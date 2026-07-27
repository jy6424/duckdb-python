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


def read_grid_mapping(con, grid_path):
    columns = con.execute(
        """
        SELECT
            grid::BIGINT AS grid_key,
            lats::DOUBLE AS lat
        FROM read_parquet('{}')
        """.format(
            escape_sql_string(grid_path)
        )
    ).fetchnumpy()
    grids = np.asarray(columns["grid_key"], dtype=np.int64)
    lats = np.asarray(columns["lat"], dtype=np.float64)
    if grids.size == 0:
        raise RuntimeError("empty grid file: {}".format(grid_path))

    grid_min = int(grids.min())
    grid_max = int(grids.max())
    build_size = grid_max - grid_min + 1
    if build_size <= 0:
        raise RuntimeError("invalid grid range in {}".format(grid_path))

    group_lats, inverse = np.unique(lats, return_inverse=True)
    grid_to_group = np.full(build_size, -1, dtype=np.int32)
    grid_offsets = (grids - grid_min).astype(np.int64, copy=False)
    grid_to_group[grid_offsets] = inverse.astype(np.int32, copy=False)

    return grid_min, grid_max, grid_to_group, group_lats.tolist()


def read_probe_columns(con, parquet_path, var):
    columns = con.execute(
        """
        SELECT
            grid::BIGINT AS grid_key,
            COALESCE({var}, 0)::DOUBLE AS value,
            CASE WHEN {var} IS NULL THEN 0 ELSE 1 END::UTINYINT AS value_valid
        FROM read_parquet('{path}')
        """.format(
            var='"' + var.replace('"', '""') + '"',
            path=escape_sql_string(parquet_path),
        )
    ).fetchnumpy()

    grids = np.asarray(columns["grid_key"], dtype=np.int64)
    values = np.asarray(columns["value"], dtype=np.float64)
    validity = np.asarray(columns["value_valid"], dtype=np.uint8)
    return grids, values, validity


def main():
    parser = argparse.ArgumentParser(description="Experimental fused GPU join+lat aggregate path")
    parser.add_argument("base_dir")
    parser.add_argument("--var", default="qicps")
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

    parquet_paths = sorted(glob.glob(os.path.join(args.base_dir, "UP-*", "time-levs-grid.parquet")))
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
        grid_path = os.path.join(os.path.dirname(parquet_path), "grid.parquet")

        grid_min, grid_max, grid_to_group, group_lats = read_grid_mapping(con, grid_path)
        grids, values, validity = read_probe_columns(con, parquet_path, args.var)

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
        print("\n[Avg {} by latitude across Parquet files]".format(args.var))
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
