import argparse
import glob
import time

import numpy as np
import xarray as xr


DEFAULT_VARIABLES = (
    "T",
    "u",
    "v",
    "qv",
    "hgt",
    "omega",
    "rh",
    "cld",
    "qc",
    "q",
)

DROP_VARIABLES = (
    "soil_levs_bnds",
    "time_bnds",
    "time_ini_bnds",
)


def tic():
    return time.perf_counter()


def parse_vars(value):
    if value == "default":
        return DEFAULT_VARIABLES
    return tuple(v.strip() for v in value.split(",") if v.strip())


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
    raise ValueError("unknown activation: {}".format(activation))


def normalize_tensor_numpy(data, eps, activation):
    norm_start = tic()

    stats_start = tic()
    mean = np.nanmean(data, axis=(0, 1), keepdims=True)
    std = np.nanstd(data, axis=(0, 1), keepdims=True)
    stats_time = tic() - stats_start

    apply_start = tic()
    normalized = ((data - mean) / (std + eps)).astype(np.float32)
    activation_start = tic()
    normalized = apply_activation_numpy(normalized, activation)
    activation_time = tic() - activation_start
    apply_time = tic() - apply_start

    norm_time = tic() - norm_start

    print("\n[Normalization]")
    print("backend: numpy")
    print("activation: {}".format(activation))
    print("stats_mean_std: {:.6f}s".format(stats_time))
    print("apply_normalization: {:.6f}s".format(apply_time))
    print("apply_activation: {:.6f}s".format(activation_time))
    print("normalization_total: {:.6f}s".format(norm_time))
    print("mean_shape: {}".format(mean.shape))
    print("std_shape: {}".format(std.shape))

    return normalized, mean, std, norm_time


def activate_tensor_numpy(data, activation):
    activation_start = tic()
    activated = apply_activation_numpy(data, activation)
    activation_time = tic() - activation_start

    print("\n[Activation]")
    print("backend: numpy")
    print("activation: {}".format(activation))
    print("activation_total: {:.6f}s".format(activation_time))

    return activated, activation_time


def read_single_file(nc_path, variables, grid_start=0, grid_count=15002):
    total_start = tic()
    profiles = []

    open_start = tic()
    with xr.open_dataset(
        nc_path,
        engine="netcdf4",
        decode_times=False,
        drop_variables=list(DROP_VARIABLES),
    ) as dataset:
        open_time = tic() - open_start

        grid_slice = slice(grid_start, grid_start + grid_count)

        print("\n==================================================")
        print("[File] {}".format(nc_path))
        print("==================================================")
        print("open_dataset: {:.6f}s".format(open_time))

        for name in variables:
            var_start = tic()

            check_start = tic()
            if name not in dataset.data_vars:
                raise KeyError("{}에 {} 변수가 없습니다.".format(nc_path, name))
            check_time = tic() - check_start

            select_start = tic()
            array = dataset[name]
            selected = array.isel(time=0, grid=grid_slice).transpose("grid", "levs")
            select_time = tic() - select_start

            load_start = tic()
            values = np.asarray(selected.values, dtype=np.float32)
            load_time = tic() - load_start

            clean_start = tic()
            values[~np.isfinite(values)] = np.nan
            values[np.abs(values) > 1.0e30] = np.nan
            clean_time = tic() - clean_start

            append_start = tic()
            profiles.append(values)
            append_time = tic() - append_start

            var_total = tic() - var_start

            print("\n[{}]".format(name))
            print("check_variable: {:.6f}s".format(check_time))
            print("select_transpose: {:.6f}s".format(select_time))
            print("load_values_to_numpy: {:.6f}s".format(load_time))
            print("clean_values: {:.6f}s".format(clean_time))
            print("append_profile: {:.6f}s".format(append_time))
            print("variable_total: {:.6f}s".format(var_total))
            print("shape: {}".format(values.shape))

        levels_start = tic()
        levels = np.asarray(dataset["levs"].values, dtype=np.float64)
        levels_time = tic() - levels_start

    stack_start = tic()
    stacked = np.stack(profiles, axis=1).astype(np.float32)
    stack_time = tic() - stack_start

    total_time = tic() - total_start

    print("\n[File Summary]")
    print("load_levels: {:.6f}s".format(levels_time))
    print("stack_profiles: {:.6f}s".format(stack_time))
    print("file_total: {:.6f}s".format(total_time))
    print("stacked_shape: {}".format(stacked.shape))
    print("levels_shape: {}".format(levels.shape))

    return stacked, levels, total_time


def read_training_data(nc_paths, variables, grid_start=0, grid_count=15002):
    total_start = tic()

    all_files = []
    reference_levels = None
    file_times = []

    print("[Number of files] {}".format(len(nc_paths)))
    print("[Variable count] {}".format(len(variables)))
    print("[Variables] {}".format(",".join(variables)))

    for i, nc_path in enumerate(nc_paths):
        print("\n##################################################")
        print("File {}/{}".format(i + 1, len(nc_paths)))
        print("##################################################")

        stacked, levels, file_time = read_single_file(
            nc_path,
            variables,
            grid_start=grid_start,
            grid_count=grid_count,
        )

        if reference_levels is None:
            reference_levels = levels
        elif not np.array_equal(reference_levels, levels):
            raise ValueError("{}의 levs 값이 이전 파일과 다릅니다.".format(nc_path))

        all_files.append(stacked)
        file_times.append(file_time)

    final_stack_start = tic()
    all_data = np.stack(all_files, axis=0).astype(np.float32)
    final_stack_time = tic() - final_stack_start

    total_time = tic() - total_start

    return all_data, reference_levels, file_times, final_stack_time, total_time


def main():
    parser = argparse.ArgumentParser(
        description="Read multiple NetCDF files into a NumPy training tensor."
    )
    parser.add_argument("--nc-pattern", default="/home/jiwan/UP-*.nc")
    parser.add_argument("--vars", default="default")
    parser.add_argument("--grid-start", type=int, default=0)
    parser.add_argument("--grid-count", type=int, default=15002)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--eps", type=float, default=1.0e-6)
    parser.add_argument(
        "--activation",
        default="none",
        choices=("none", "sigmoid", "relu", "tanh", "gelu", "softplus"),
    )
    args = parser.parse_args()

    variables = parse_vars(args.vars)
    nc_paths = sorted(glob.glob(args.nc_pattern))
    if args.limit > 0:
        nc_paths = nc_paths[: args.limit]
    if not nc_paths:
        raise FileNotFoundError("no files matched: {}".format(args.nc_pattern))

    all_data, levels, file_times, final_stack_time, read_total_time = read_training_data(
        nc_paths,
        variables,
        grid_start=args.grid_start,
        grid_count=args.grid_count,
    )

    transform_time = 0.0
    if args.normalize:
        all_data, _, _, transform_time = normalize_tensor_numpy(
            all_data,
            eps=args.eps,
            activation=args.activation,
        )
    elif args.activation != "none":
        all_data, transform_time = activate_tensor_numpy(all_data, args.activation)

    total_time = read_total_time + transform_time

    print("\n==================================================")
    print("[Overall Summary]")
    print("==================================================")
    print("number_of_files: {}".format(len(nc_paths)))
    print("variable_count: {}".format(len(variables)))
    print("sum_file_times: {:.6f}s".format(sum(file_times)))
    print("avg_file_time: {:.6f}s".format(sum(file_times) / len(file_times)))
    print("min_file_time: {:.6f}s".format(min(file_times)))
    print("max_file_time: {:.6f}s".format(max(file_times)))
    print("final_stack_time: {:.6f}s".format(final_stack_time))
    print("read_total_time: {:.6f}s".format(read_total_time))
    print("transform_time: {:.6f}s".format(transform_time))
    print("total_time: {:.6f}s".format(total_time))
    print("final_shape: {}".format(all_data.shape))
    print("levels_shape: {}".format(levels.shape))

    print("\n[Per-file times]")
    for i, (path, elapsed) in enumerate(zip(nc_paths, file_times)):
        print("{}: {:.6f}s {}".format(i + 1, elapsed, path))


if __name__ == "__main__":
    main()
