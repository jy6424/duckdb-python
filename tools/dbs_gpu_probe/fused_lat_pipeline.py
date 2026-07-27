#!/usr/bin/env python3
import argparse
import ctypes
import glob
import os
import time

import duckdb
import numpy as np


def escape_sql_string(value):
    return value.replace("'", "''")


def quote_ident(name):
    return '"' + name.replace('"', '""') + '"'


def bind_fused_function(function):
    function.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS"),
        ctypes.c_uint64,
        ctypes.c_int64,
        ctypes.c_int64,
        np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
        ctypes.c_uint64,
        ctypes.c_uint64,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS"),
    ]
    function.restype = ctypes.c_int


def load_gpu_lib(path):
    lib = ctypes.CDLL(path)
    bind_fused_function(lib.duckdb_gpu_fused_lat_agg_i64_double)
    bind_fused_function(lib.duckdb_gpu_fused_lat_agg_i64_double_mapped)
    return lib


def read_group_mapping(con, dimension_path, join_key, group_column):
    columns = con.execute(
        """
        SELECT
            {join_key}::BIGINT AS join_key,
            {group_column}::DOUBLE AS group_value
        FROM read_parquet('{}')
        """.format(
            escape_sql_string(dimension_path),
            join_key=quote_ident(join_key),
            group_column=quote_ident(group_column),
        )
    ).fetchnumpy()
    join_keys = np.asarray(columns["join_key"], dtype=np.int64)
    group_values = np.asarray(columns["group_value"], dtype=np.float64)
    if join_keys.size == 0:
        raise RuntimeError("empty dimension file: {}".format(dimension_path))

    join_min = int(join_keys.min())
    join_max = int(join_keys.max())
    build_size = join_max - join_min + 1
    if build_size <= 0:
        raise RuntimeError("invalid join key range in {}".format(dimension_path))

    groups, inverse = np.unique(group_values, return_inverse=True)
    grid_to_group = np.full(build_size, -1, dtype=np.int32)
    join_offsets = (join_keys - join_min).astype(np.int64, copy=False)
    grid_to_group[join_offsets] = inverse.astype(np.int32, copy=False)

    return join_min, join_max, grid_to_group, groups.tolist()


def read_probe_columns(con, parquet_path, join_key, var):
    columns = con.execute(
        """
        SELECT
            {join_key}::BIGINT AS join_key,
            COALESCE({var}, 0)::DOUBLE AS value,
            CASE WHEN {var} IS NULL THEN 0 ELSE 1 END::UTINYINT AS value_valid
        FROM read_parquet('{path}')
        """.format(
            join_key=quote_ident(join_key),
            var=quote_ident(var),
            path=escape_sql_string(parquet_path),
        )
    ).fetchnumpy()

    grids = np.asarray(columns["join_key"], dtype=np.int64)
    values = np.asarray(columns["value"], dtype=np.float64)
    validity = np.asarray(columns["value_valid"], dtype=np.uint8)
    return grids, values, validity


def main():
    parser = argparse.ArgumentParser(description="Experimental fused GPU join+lat aggregate path")
    parser.add_argument("base_dir")
    parser.add_argument("--var", default="qicps")
    parser.add_argument("--join-key", default="grid")
    parser.add_argument("--group-column", default="lats")
    parser.add_argument("--fact-file", default="time-levs-grid.parquet")
    parser.add_argument("--dimension-file", default="grid.parquet")
    parser.add_argument("--max-files", type=int, default=0, help="Process only the first N files when N > 0")
    parser.add_argument(
        "--mode",
        choices=["device", "mapped"],
        default="device",
        help="device uses cudaMemcpy HtoD; mapped uses cudaHostAllocMapped zero-copy inputs",
    )
    parser.add_argument("--print-averages", action="store_true")
    parser.add_argument(
        "--lib",
        default=os.path.join(os.path.dirname(__file__), "libduckdb_gpu_probe.so"),
        help="Path to libduckdb_gpu_probe.so",
    )
    args = parser.parse_args()

    parquet_paths = sorted(glob.glob(os.path.join(args.base_dir, "UP-*", args.fact_file)))
    if args.max_files > 0:
        parquet_paths = parquet_paths[: args.max_files]
    if not parquet_paths:
        raise RuntimeError("no input parquet files found below {}".format(args.base_dir))

    gpu = load_gpu_lib(args.lib)
    fused_agg = (
        gpu.duckdb_gpu_fused_lat_agg_i64_double_mapped
        if args.mode == "mapped"
        else gpu.duckdb_gpu_fused_lat_agg_i64_double
    )
    con = duckdb.connect()

    total_sum = {}
    total_count = {}
    total_rows = 0

    start = time.time()
    for parquet_path in parquet_paths:
        print("reading: {}".format(parquet_path))
        grid_path = os.path.join(os.path.dirname(parquet_path), args.dimension_file)

        grid_min, grid_max, grid_to_group, group_lats = read_group_mapping(
            con, grid_path, args.join_key, args.group_column
        )
        grids, values, validity = read_probe_columns(con, parquet_path, args.join_key, args.var)

        group_count = len(group_lats)
        sums = np.zeros(group_count, dtype=np.float64)
        counts = np.zeros(group_count, dtype=np.uint64)
        row_counts = np.zeros(group_count, dtype=np.uint64)

        rc = fused_agg(
            grids,
            values,
            validity,
            grids.size,
            grid_min,
            grid_max,
            grid_to_group,
            grid_to_group.size,
            group_count,
            sums,
            counts,
            row_counts,
        )
        if rc != 0:
            raise RuntimeError("GPU fused lat aggregate failed for {}".format(parquet_path))

        for group, lat in enumerate(group_lats):
            row_count = int(row_counts[group])
            if row_count == 0:
                continue
            lat_key = float(lat)
            total_sum[lat_key] = total_sum.get(lat_key, 0.0) + float(sums[group])
            total_count[lat_key] = total_count.get(lat_key, 0) + int(counts[group])
            total_rows += row_count

    elapsed = time.time() - start

    print("\n[Row Count]")
    print(total_rows)

    if args.print_averages:
        print("\n[Avg {} by {} across Parquet files]".format(args.var, args.group_column))
        for lat in sorted(total_sum.keys()):
            count = total_count[lat]
            if count == 0:
                continue
            avg = total_sum[lat] / count
            print("({:.17g}, {:.17g})".format(float(lat), float(avg)))

    print("[number of input file]: {}".format(len(parquet_paths)))
    print("[query time]: {:.6f}s".format(elapsed))
    con.close()


if __name__ == "__main__":
    main()
