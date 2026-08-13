#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import glob
import os
import time

import duckdb


EXCLUDED_AUTO_COLUMNS = set(["grid", "time", "levs"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir")
    parser.add_argument("--vars", default="all")
    parser.add_argument("--join-key", default="grid")
    parser.add_argument("--group-column", default="lats")
    parser.add_argument("--dimension-file", default="grid.parquet")
    parser.add_argument("--read-mode", default="glob", choices=["glob", "per-file"])
    parser.add_argument("--mode", default="device", choices=["device", "mapped"])
    parser.add_argument(
        "--lib",
        default=os.environ.get(
            "DUCKDB_GPU_PROBE_LIB",
            "/home/jiwan/duckdb-python/tools/dbs_gpu_probe/libduckdb_gpu_probe.so",
        ),
    )
    parser.add_argument("--print-results", action="store_true")
    return parser.parse_args()


def read_auto_payload_columns(parquet_path):
    con = duckdb.connect()
    try:
        rows = con.execute("""
            DESCRIBE SELECT *
            FROM read_parquet(?)
        """, [parquet_path]).fetchall()
    finally:
        con.close()

    columns = []
    for name, typ, *_ in rows:
        if typ.upper() == "DOUBLE" and name not in EXCLUDED_AUTO_COLUMNS:
            columns.append(name)
    return columns


def read_proc_io():
    data = {}
    with open("/proc/self/io", "r") as f:
        for line in f:
            k, v = line.strip().split(":")
            data[k.strip()] = int(v.strip())
    return data


def write_io_diff(path, title, before, after, elapsed):
    with open(path, "a") as f:
        f.write("\n=== {} ===\n".format(title))
        f.write("[query time]: {:.6f}s\n".format(elapsed))
        for k in sorted(after.keys()):
            f.write("{}: {} -> {} diff={}\n".format(k, before.get(k, 0), after[k], after[k] - before.get(k, 0)))


def main():
    args = parse_args()
    parquet_paths = sorted(glob.glob(os.path.join(args.base_dir, "UP-*", "time-levs-grid.parquet")))
    if not parquet_paths:
        raise SystemExit("no time-levs-grid.parquet files found")

    grid_paths = sorted(glob.glob(os.path.join(args.base_dir, "UP-*", args.dimension_file)))
    if not grid_paths:
        raise SystemExit("no {} files found".format(args.dimension_file))

    if args.vars.strip().lower() == "all":
        payload_columns = read_auto_payload_columns(parquet_paths[0])
    else:
        payload_columns = [column.strip() for column in args.vars.split(",") if column.strip()]
    if not payload_columns:
        raise SystemExit("no payload columns specified")

    if args.read_mode == "glob":
        fact_inputs = [os.path.join(args.base_dir, "UP-*", "time-levs-grid.parquet")]
        dimension_file = grid_paths[0]
    else:
        fact_inputs = parquet_paths
        dimension_file = args.dimension_file

    before_io = read_proc_io()
    start = time.time()

    result = duckdb.dbs_gpu_fused_lat_multi(
        fact_inputs,
        payload_columns=payload_columns,
        join_key=args.join_key,
        group_column=args.group_column,
        dimension_file=dimension_file,
        lib_path=args.lib,
        mode=args.mode,
    )

    elapsed = time.time() - start
    after_io = read_proc_io()
    write_io_diff("io_duckdb_direct_gpu.txt", "case 4 direct gpu fused aggregate", before_io, after_io, elapsed)

    print("\n[Row Count]")
    print(result["row_count"])
    print("[number of input file]: {}".format(len(parquet_paths)))
    print("[read mode]: {}".format(args.read_mode))
    print("[gpu mode]: {}".format(args.mode))
    print("[payload columns]: {}".format(",".join(result["payload_columns"])))
    print("[query time]: {:.6f}s".format(float(result["query_time"])))
    print("[wrapper time]: {:.6f}s".format(elapsed))

    if args.print_results:
        print("\n[Results]")
        for row in result["groups"]:
            values = ["group={:.17g}".format(float(row["group"])), "row_count={}".format(int(row["row_count"]))]
            for column in result["payload_columns"]:
                values.append("sum_{}={:.17g}".format(column, float(row["sum_" + column])))
                values.append("count_{}={}".format(column, int(row["count_" + column])))
            print(", ".join(values))


if __name__ == "__main__":
    main()
