import argparse
import glob
import os
import time

import duckdb
import numpy as np


DEFAULT_VARIABLES = (
    "T",
    "u",
    "v",
    "qv",
    "hgt",
    "p",
)

EXCLUDED_COLUMNS = {
    "time",
    "levs",
    "grid",
}


def tic():
    return time.perf_counter()


def get_all_double_variables(con, parquet_path):
    rows = con.execute(
        """
        DESCRIBE SELECT *
        FROM read_parquet(
            ?,
            union_by_name=false,
            hive_partitioning=false,
            filename=false,
            file_row_number=false,
            binary_as_string=false
        )
        """,
        [parquet_path],
    ).fetchall()

    variables = []
    for name, typ, *_ in rows:
        if name not in EXCLUDED_COLUMNS and typ.upper() in ("DOUBLE", "FLOAT", "REAL"):
            variables.append(name)
    return variables


def build_clean_expr(name):
    return (
        f"CASE "
        f"WHEN isfinite({name}) "
        f"AND abs({name}) <= 1.0e30 "
        f"THEN CAST({name} AS FLOAT) "
        f"ELSE NULL "
        f"END AS {name}"
    )


def apply_activation_numpy(data, activation):
    if activation == "none":
        return data
    if activation == "sigmoid":
        return (1.0 / (1.0 + np.exp(-data))).astype(np.float32)
    if activation == "relu":
        return np.maximum(data, 0.0).astype(np.float32)
    if activation == "tanh":
        return np.tanh(data).astype(np.float32)
    if activation == "gelu":
        cubed = data * data * data
        return (
            0.5
            * data
            * (1.0 + np.tanh(0.7978845608028654 * (data + 0.044715 * cubed)))
        ).astype(np.float32)
    if activation == "softplus":
        return np.where(
            data > 0.0,
            data + np.log1p(np.exp(-data)),
            np.log1p(np.exp(data)),
        ).astype(np.float32)
    raise ValueError(f"unknown normalize activation: {activation}")


def apply_activation_cupy(data, activation, cp):
    if activation == "none":
        return data
    if activation == "sigmoid":
        return (1.0 / (1.0 + cp.exp(-data))).astype(cp.float32)
    if activation == "relu":
        return cp.maximum(data, 0.0).astype(cp.float32)
    if activation == "tanh":
        return cp.tanh(data).astype(cp.float32)
    if activation == "gelu":
        cubed = data * data * data
        return (
            0.5
            * data
            * (1.0 + cp.tanh(0.7978845608028654 * (data + 0.044715 * cubed)))
        ).astype(cp.float32)
    if activation == "softplus":
        return cp.where(
            data > 0.0,
            data + cp.log1p(cp.exp(-data)),
            cp.log1p(cp.exp(data)),
        ).astype(cp.float32)
    raise ValueError(f"unknown normalize activation: {activation}")


def normalize_tensor_numpy(data, eps=1.0e-6, activation="none"):
    norm_start = tic()

    stats_start = tic()
    mean = np.nanmean(data, axis=(0, 1), keepdims=True)
    std = np.nanstd(data, axis=(0, 1), keepdims=True)
    stats_time = tic() - stats_start

    apply_start = tic()
    normalized = ((data - mean) / (std + eps)).astype(np.float32)
    normalized = apply_activation_numpy(normalized, activation)
    apply_time = tic() - apply_start

    norm_time = tic() - norm_start

    print("\n[Normalization]")
    print("backend: numpy")
    print(f"activation: {activation}")
    print(f"stats_mean_std: {stats_time:.6f}s")
    print(f"apply_normalization: {apply_time:.6f}s")
    print(f"normalization_total: {norm_time:.6f}s")
    print(f"mean_shape: {mean.shape}")
    print(f"std_shape: {std.shape}")

    return normalized, mean, std, norm_time


def normalize_tensor_cupy(data, eps=1.0e-6, return_gpu=False, activation="none"):
    import cupy as cp

    norm_start = tic()

    copy_start = tic()
    gpu_data = cp.asarray(data)
    cp.cuda.Stream.null.synchronize()
    copy_to_gpu_time = tic() - copy_start

    stats_start = tic()
    mean = cp.nanmean(gpu_data, axis=(0, 1), keepdims=True)
    std = cp.nanstd(gpu_data, axis=(0, 1), keepdims=True)
    cp.cuda.Stream.null.synchronize()
    stats_time = tic() - stats_start

    apply_start = tic()
    normalized = ((gpu_data - mean) / (std + eps)).astype(cp.float32)
    normalized = apply_activation_cupy(normalized, activation, cp)
    cp.cuda.Stream.null.synchronize()
    apply_time = tic() - apply_start

    if return_gpu:
        normalized_out = normalized
        mean_out = mean
        std_out = std
        copy_back_time = 0.0
    else:
        copy_back_start = tic()
        normalized_out = cp.asnumpy(normalized)
        mean_out = cp.asnumpy(mean)
        std_out = cp.asnumpy(std)
        copy_back_time = tic() - copy_back_start

    norm_time = tic() - norm_start

    print("\n[Normalization]")
    print("backend: cupy")
    print(f"activation: {activation}")
    print(f"copy_to_gpu: {copy_to_gpu_time:.6f}s")
    print(f"stats_mean_std: {stats_time:.6f}s")
    print(f"apply_normalization: {apply_time:.6f}s")
    print(f"copy_back_cpu: {copy_back_time:.6f}s")
    print(f"normalization_total: {norm_time:.6f}s")
    print(f"mean_shape: {mean.shape}")
    print(f"std_shape: {std.shape}")

    return normalized_out, mean_out, std_out, norm_time


def normalize_tensor_duckdb_gpu(data, eps=1.0e-6, lib_path="", activation="none"):
    norm_start = tic()

    input_start = tic()
    gpu_input = np.ascontiguousarray(data, dtype=np.float32)
    input_time = tic() - input_start

    call_start = tic()
    old_activation = os.environ.get("DUCKDB_GPU_NORMALIZE_ACTIVATION")
    os.environ["DUCKDB_GPU_NORMALIZE_ACTIVATION"] = activation
    try:
        result = duckdb.dbs_gpu_normalize_tensor(gpu_input, lib_path=lib_path, eps=eps)
    finally:
        if old_activation is None:
            os.environ.pop("DUCKDB_GPU_NORMALIZE_ACTIVATION", None)
        else:
            os.environ["DUCKDB_GPU_NORMALIZE_ACTIVATION"] = old_activation
    call_time = tic() - call_start

    normalized = result["normalized"]
    mean = result["mean"]
    std = result["std"]
    norm_time = tic() - norm_start

    print("\n[Normalization]")
    print("backend: duckdb-gpu")
    print(f"activation: {activation}")
    print(f"prepare_contiguous_input: {input_time:.6f}s")
    print(f"gpu_call_wall: {call_time:.6f}s")
    print(f"gpu_kernel_total: {result['normalization_time']:.6f}s")
    print(f"normalization_total: {norm_time:.6f}s")
    print(f"mean_shape: {mean.shape}")
    print(f"std_shape: {std.shape}")

    return normalized, mean, std, norm_time


def normalize_tensor_duckdb_gpu_direct(
    paths,
    variables,
    grid_start=0,
    grid_count=15002,
    eps=1.0e-6,
    lib_path="",
    activation="none",
):
    norm_start = tic()

    old_activation = os.environ.get("DUCKDB_GPU_NORMALIZE_ACTIVATION")
    os.environ["DUCKDB_GPU_NORMALIZE_ACTIVATION"] = activation
    try:
        result = duckdb.dbs_gpu_read_normalize_tensor(
            paths,
            tuple(variables),
            grid_start=grid_start,
            grid_count=grid_count,
            lib_path=lib_path,
            eps=eps,
        )
    finally:
        if old_activation is None:
            os.environ.pop("DUCKDB_GPU_NORMALIZE_ACTIVATION", None)
        else:
            os.environ["DUCKDB_GPU_NORMALIZE_ACTIVATION"] = old_activation

    norm_time = tic() - norm_start

    print("\n[Normalization]")
    print("backend: duckdb-gpu-direct")
    print(f"activation: {activation}")
    print(f"direct_read_time: {result['direct_read_time']:.6f}s")
    print(f"gpu_kernel_total: {result['normalization_time']:.6f}s")
    print(f"normalization_total: {norm_time:.6f}s")
    print(f"rows_scanned: {result['rows_scanned']}")
    print(f"rows_selected: {result['rows_selected']}")
    print(f"scan_calls: {result['scan_calls']}")
    print(f"mean_shape: {result['mean'].shape}")
    print(f"std_shape: {result['std'].shape}")

    return result["normalized"], result["mean"], result["std"], norm_time, result


def read_training_data_duckdb(
    parquet_path,
    variables=DEFAULT_VARIABLES,
    grid_start=0,
    grid_count=15002,
    use_all_variables=False,
):
    total_start = tic()

    connect_start = tic()
    con = duckdb.connect()
    connect_time = tic() - connect_start

    try:
        discover_start = tic()
        if use_all_variables:
            variables = get_all_double_variables(con, parquet_path)
        else:
            variables = list(variables)
        discover_time = tic() - discover_start

        query_start = tic()
        value_exprs = [build_clean_expr(name) for name in variables]

        query = f"""
        SELECT
            grid,
            levs,
            {", ".join(value_exprs)}
        FROM read_parquet(
            ?,
            union_by_name=false,
            hive_partitioning=false,
            filename=false,
            file_row_number=false,
            binary_as_string=false
        )
        WHERE grid >= ?
          AND grid < ?
        """
        query_build_time = tic() - query_start

        execute_start = tic()
        result = con.execute(query, [parquet_path, grid_start, grid_start + grid_count])
        execute_time = tic() - execute_start

        fetch_start = tic()
        arrays = result.fetchnumpy()
        fetch_time = tic() - fetch_start

        if len(arrays["grid"]) == 0:
            raise ValueError(f"No rows returned from {parquet_path}")

        levels_start = tic()
        levels = np.unique(arrays["levs"]).astype(np.float64)
        levels_time = tic() - levels_start

        shape_start = tic()
        ngrid = len(np.unique(arrays["grid"]))
        nlev = len(levels)
        nvar = len(variables)
        expected_rows = ngrid * nlev
        actual_rows = len(arrays["grid"])
        if actual_rows != expected_rows:
            raise ValueError(
                f"Unexpected row count in {parquet_path}: "
                f"actual={actual_rows}, expected={expected_rows}"
            )
        shape_time = tic() - shape_start

        stack_start = tic()
        columns = [np.asarray(arrays[name], dtype=np.float32) for name in variables]
        values = np.stack(columns, axis=1)
        stack_time = tic() - stack_start

        reshape_start = tic()
        stacked = values.reshape(ngrid, nlev, nvar).transpose(0, 2, 1)
        reshape_time = tic() - reshape_start

    finally:
        close_start = tic()
        con.close()
        close_time = tic() - close_start

    file_total = tic() - total_start

    print("\n==========================================")
    print(f"[Parquet] {parquet_path}")
    print("==========================================")
    print(f"variable_count: {len(variables)}")
    print(f"grid_start: {grid_start}")
    print(f"grid_count: {grid_count}")

    print("\n[Phase Times]")
    print(f"connect: {connect_time:.6f}s")
    print(f"discover_variables: {discover_time:.6f}s")
    print(f"query_build: {query_build_time:.6f}s")
    print(f"execute: {execute_time:.6f}s")
    print(f"fetch_to_numpy: {fetch_time:.6f}s")
    print(f"load_levels: {levels_time:.6f}s")
    print(f"infer_shape: {shape_time:.6f}s")
    print(f"stack_columns: {stack_time:.6f}s")
    print(f"reshape_transpose: {reshape_time:.6f}s")
    print(f"close: {close_time:.6f}s")
    print(f"file_total: {file_total:.6f}s")

    print("\n[Shapes]")
    print(f"stacked_shape: {stacked.shape}")
    print(f"levels_shape: {levels.shape}")

    return stacked, levels, tuple(variables), file_total


def read_many_training_files_duckdb(
    parquet_pattern="/home/jiwan/UP-*/time-levs-grid.parquet",
    variables=DEFAULT_VARIABLES,
    grid_start=0,
    grid_count=15002,
    limit=None,
    use_all_variables=False,
    normalize=False,
    normalize_backend="numpy",
    normalize_return_gpu=False,
    gpu_lib_path="",
    normalize_activation="none",
):
    total_start = tic()

    paths = sorted(glob.glob(parquet_pattern))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No Parquet files found: {parquet_pattern}")

    if use_all_variables:
        con = duckdb.connect()
        try:
            variables = tuple(get_all_double_variables(con, paths[0]))
        finally:
            con.close()
    else:
        variables = tuple(variables)

    if normalize and normalize_backend == "duckdb-gpu-direct":
        data, mean, std, normalize_time, direct_result = normalize_tensor_duckdb_gpu_direct(
            paths,
            variables,
            grid_start=grid_start,
            grid_count=grid_count,
            lib_path=gpu_lib_path,
            activation=normalize_activation,
        )
        total_time = tic() - total_start

        print("\n==================================================")
        print("[Overall Summary]")
        print("==================================================")
        print(f"number_of_files: {len(paths)}")
        print(f"variable_count: {len(variables)}")
        print("sum_file_times: 0.000000s")
        print("avg_file_time: 0.000000s")
        print("min_file_time: 0.000000s")
        print("max_file_time: 0.000000s")
        print("final_stack_time: 0.000000s")
        print(f"normalization_time: {normalize_time:.6f}s")
        print(f"total_time: {total_time:.6f}s")
        print(f"final_shape: {data.shape}")
        print(f"levels_shape: ({direct_result['level_count']},)")
        return data, None, variables, mean, std

    all_stacked = []
    levels_ref = None
    variables_ref = None
    file_times = []

    for i, path in enumerate(paths, 1):
        file_start = tic()
        stacked, levels, used_variables, measured_file_time = read_training_data_duckdb(
            path,
            variables=variables,
            grid_start=grid_start,
            grid_count=grid_count,
            use_all_variables=use_all_variables,
        )
        elapsed = tic() - file_start

        all_stacked.append(stacked)
        file_times.append(elapsed)

        if levels_ref is None:
            levels_ref = levels
        elif not np.array_equal(levels_ref, levels):
            raise ValueError(f"levels differ: {path}")

        if variables_ref is None:
            variables_ref = tuple(used_variables)
        elif variables_ref != tuple(used_variables):
            raise ValueError(f"variables differ: {path}")

        print(f"\n[Progress] {i}/{len(paths)}")
        print(f"measured_file_time: {measured_file_time:.6f}s")
        print(f"outer_file_time: {elapsed:.6f}s")

    final_stack_start = tic()
    data = np.stack(all_stacked, axis=0)
    final_stack_time = tic() - final_stack_start

    mean = None
    std = None
    normalize_time = 0.0
    if normalize:
        if normalize_backend == "numpy":
            data, mean, std, normalize_time = normalize_tensor_numpy(
                data,
                activation=normalize_activation,
            )
        elif normalize_backend == "cupy":
            data, mean, std, normalize_time = normalize_tensor_cupy(
                data,
                return_gpu=normalize_return_gpu,
                activation=normalize_activation,
            )
        elif normalize_backend == "duckdb-gpu":
            data, mean, std, normalize_time = normalize_tensor_duckdb_gpu(
                data,
                lib_path=gpu_lib_path,
                activation=normalize_activation,
            )
        else:
            raise ValueError(f"unknown normalize_backend: {normalize_backend}")

    total_time = tic() - total_start

    print("\n==================================================")
    print("[Overall Summary]")
    print("==================================================")
    print(f"number_of_files: {len(paths)}")
    print(f"variable_count: {len(variables_ref)}")
    print(f"sum_file_times: {sum(file_times):.6f}s")
    print(f"avg_file_time: {np.mean(file_times):.6f}s")
    print(f"min_file_time: {np.min(file_times):.6f}s")
    print(f"max_file_time: {np.max(file_times):.6f}s")
    print(f"final_stack_time: {final_stack_time:.6f}s")
    print(f"normalization_time: {normalize_time:.6f}s")
    print(f"total_time: {total_time:.6f}s")
    print(f"final_shape: {data.shape}")
    print(f"levels_shape: {levels_ref.shape}")

    print("\n[Per-file times]")
    for i, (path, elapsed) in enumerate(zip(paths, file_times), 1):
        print(f"{i}: {elapsed:.6f}s {path}")

    if normalize:
        return data, levels_ref, variables_ref, mean, std
    return data, levels_ref, variables_ref


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet-pattern",
        default="/home/jiwan/UP-*/time-levs-grid.parquet",
    )
    parser.add_argument("--vars", default=",".join(DEFAULT_VARIABLES))
    parser.add_argument("--all-vars", action="store_true")
    parser.add_argument("--grid-start", type=int, default=0)
    parser.add_argument("--grid-count", type=int, default=15002)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument(
        "--normalize-backend",
        choices=["numpy", "cupy", "duckdb-gpu", "duckdb-gpu-direct"],
        default="numpy",
    )
    parser.add_argument(
        "--normalize-activation",
        choices=["none", "sigmoid", "relu", "tanh", "gelu", "softplus"],
        default="none",
    )
    parser.add_argument("--gpu-lib-path", default="")
    parser.add_argument(
        "--normalize-return-gpu",
        action="store_true",
        help="keep normalized output as a CuPy array when --normalize-backend=cupy",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    variables = tuple(v.strip() for v in args.vars.split(",") if v.strip())

    read_many_training_files_duckdb(
        parquet_pattern=args.parquet_pattern,
        variables=variables,
        grid_start=args.grid_start,
        grid_count=args.grid_count,
        limit=args.limit,
        use_all_variables=args.all_vars,
        normalize=args.normalize,
        normalize_backend=args.normalize_backend,
        normalize_return_gpu=args.normalize_return_gpu,
        gpu_lib_path=args.gpu_lib_path,
        normalize_activation=args.normalize_activation,
    )
