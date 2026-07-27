#!/usr/bin/env python3
import argparse
import ctypes
import glob
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import duckdb
import numpy as np


_thread_local = threading.local()


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


def get_thread_connection():
    con = getattr(_thread_local, "connection", None)
    if con is None:
        con = duckdb.connect()
        _thread_local.connection = con
    return con


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


def read_file_inputs(parquet_path, args):
    read_start = time.time()
    con = get_thread_connection()
    grid_path = os.path.join(os.path.dirname(parquet_path), args.dimension_file)
    mapping_start = time.time()
    grid_min, grid_max, grid_to_group, group_lats = read_group_mapping(
        con, grid_path, args.join_key, args.group_column
    )
    probe_start = time.time()
    grids, values, validity = read_probe_columns(con, parquet_path, args.join_key, args.var)
    read_end = time.time()
    return {
        "path": parquet_path,
        "grid_min": grid_min,
        "grid_max": grid_max,
        "grid_to_group": grid_to_group,
        "group_lats": group_lats,
        "grids": grids,
        "values": values,
        "validity": validity,
        "mapping_time": probe_start - mapping_start,
        "probe_read_time": read_end - probe_start,
        "read_time": read_end - read_start,
    }


def run_gpu(fused_agg, item):
    group_count = len(item["group_lats"])
    sums = np.zeros(group_count, dtype=np.float64)
    counts = np.zeros(group_count, dtype=np.uint64)
    row_counts = np.zeros(group_count, dtype=np.uint64)

    gpu_start = time.time()
    rc = fused_agg(
        item["grids"],
        item["values"],
        item["validity"],
        item["grids"].size,
        item["grid_min"],
        item["grid_max"],
        item["grid_to_group"],
        item["grid_to_group"].size,
        group_count,
        sums,
        counts,
        row_counts,
    )
    gpu_time = time.time() - gpu_start
    if rc != 0:
        raise RuntimeError("GPU fused lat aggregate failed for {}".format(item["path"]))
    return sums, counts, row_counts, gpu_time


def merge_result(total_sum, total_count, item, sums, counts, row_counts):
    merge_start = time.time()
    total_rows = 0
    for group, group_value in enumerate(item["group_lats"]):
        row_count = int(row_counts[group])
        if row_count == 0:
            continue
        group_key = float(group_value)
        total_sum[group_key] = total_sum.get(group_key, 0.0) + float(sums[group])
        total_count[group_key] = total_count.get(group_key, 0) + int(counts[group])
        total_rows += row_count
    return total_rows, time.time() - merge_start


def main():
    parser = argparse.ArgumentParser(description="Pipelined experimental fused GPU join+aggregate path")
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
    parser.add_argument("--print-stage-times", action="store_true")
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

    total_sum = {}
    total_count = {}
    total_rows = 0
    total_wait_time = 0.0
    total_read_time = 0.0
    total_mapping_time = 0.0
    total_probe_read_time = 0.0
    total_gpu_time = 0.0
    total_merge_time = 0.0

    start = time.time()
    with ThreadPoolExecutor(max_workers=1) as executor:
        next_index = 0
        future = executor.submit(read_file_inputs, parquet_paths[next_index], args)
        next_index += 1

        while future is not None:
            wait_start = time.time()
            item = future.result()
            wait_time = time.time() - wait_start

            if next_index < len(parquet_paths):
                next_future = executor.submit(read_file_inputs, parquet_paths[next_index], args)
                next_index += 1
            else:
                next_future = None

            print("reading: {}".format(item["path"]))
            sums, counts, row_counts, gpu_time = run_gpu(fused_agg, item)
            row_delta, merge_time = merge_result(total_sum, total_count, item, sums, counts, row_counts)

            total_rows += row_delta
            total_wait_time += wait_time
            total_read_time += item["read_time"]
            total_mapping_time += item["mapping_time"]
            total_probe_read_time += item["probe_read_time"]
            total_gpu_time += gpu_time
            total_merge_time += merge_time

            if args.print_stage_times:
                print(
                    "[stage] wait={:.6f}s read={:.6f}s mapping={:.6f}s probe={:.6f}s gpu={:.6f}s merge={:.6f}s".format(
                        wait_time,
                        item["read_time"],
                        item["mapping_time"],
                        item["probe_read_time"],
                        gpu_time,
                        merge_time,
                    )
                )

            future = next_future

    elapsed = time.time() - start

    print("\n[Row Count]")
    print(total_rows)

    if args.print_averages:
        print("\n[Avg {} by {} across Parquet files]".format(args.var, args.group_column))
        for group_value in sorted(total_sum.keys()):
            count = total_count[group_value]
            if count == 0:
                continue
            avg = total_sum[group_value] / count
            print("({:.17g}, {:.17g})".format(float(group_value), float(avg)))

    if args.print_stage_times:
        print("\n[Stage Totals]")
        print("wait_time: {:.6f}s".format(total_wait_time))
        print("read_time: {:.6f}s".format(total_read_time))
        print("mapping_time: {:.6f}s".format(total_mapping_time))
        print("probe_read_time: {:.6f}s".format(total_probe_read_time))
        print("gpu_time: {:.6f}s".format(total_gpu_time))
        print("merge_time: {:.6f}s".format(total_merge_time))

    print("[number of input file]: {}".format(len(parquet_paths)))
    print("[query time]: {:.6f}s".format(elapsed))


if __name__ == "__main__":
    main()
