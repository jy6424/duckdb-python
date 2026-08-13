#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function

import glob
import os
import time
import argparse

import duckdb


io_log = "io_duckdb_pipeline.txt"
EXCLUDED_AUTO_COLUMNS = set(["grid", "time", "levs"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir")
    parser.add_argument("--var", default=None)
    parser.add_argument("--vars", default=None)
    parser.add_argument("--join-key", default="grid")
    parser.add_argument("--group-column", default="lats")
    parser.add_argument("--dimension-file", default="grid.parquet")
    parser.add_argument(
        "--mode",
        default=os.environ.get("DUCKDB_GPU_PIPELINE_MODE", "pipeline-device"),
        choices=["device", "mapped", "pipeline-device", "pipeline-mapped", "pipeline-managed", "pipeline-cpu"],
    )
    parser.add_argument(
        "--lib",
        default=os.environ.get(
            "DUCKDB_GPU_PROBE_LIB",
            "/home/jiwan/duckdb-python-pipeline/tools/dbs_gpu_probe/libduckdb_gpu_probe.so",
        ),
    )
    return parser.parse_args()


def read_proc_io():
    data = {}
    with open("/proc/self/io", "r") as f:
        for line in f:
            k, v = line.strip().split(":")
            data[k.strip()] = int(v.strip())
    return data


def write_io_diff(title, before, after, elapsed):
    with open(io_log, "a") as f:
        f.write("\n=== {} ===\n".format(title))
        f.write("[query time]: {:.6f}s\n".format(elapsed))
        for k in sorted(after.keys()):
            f.write(
                "{}: {} -> {} diff={}\n".format(k, before.get(k, 0), after[k], after[k] - before.get(k, 0))
            )


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


def main():
    args = parse_args()
    parquet_paths = sorted(glob.glob(os.path.join(args.base_dir, "UP-*", "time-levs-grid.parquet")))
    if not parquet_paths:
        raise SystemExit("no time-levs-grid.parquet files found")

    if args.vars and args.vars.strip().lower() == "all":
        payload_columns = read_auto_payload_columns(parquet_paths[0])
    elif args.vars:
        payload_columns = [column.strip() for column in args.vars.split(",") if column.strip()]
    else:
        payload_columns = [args.var or "qicps"]
    if not payload_columns:
        raise SystemExit("no payload columns specified")
    if len(payload_columns) > 1 and args.mode in ("pipeline-managed", "pipeline-cpu"):
        raise SystemExit("{} does not support --vars yet; use pipeline-device or pipeline-mapped".format(args.mode))

    before_io = read_proc_io()
    start = time.time()

    if len(payload_columns) == 1:
        result = duckdb.dbs_gpu_fused_lat_pipeline(
            parquet_paths,
            payload_column=payload_columns[0],
            join_key=args.join_key,
            group_column=args.group_column,
            dimension_file=args.dimension_file,
            lib_path=args.lib,
            mode=args.mode,
        )
    else:
        result = duckdb.dbs_gpu_fused_lat_multi(
            parquet_paths,
            payload_columns=payload_columns,
            join_key=args.join_key,
            group_column=args.group_column,
            dimension_file=args.dimension_file,
            lib_path=args.lib,
            mode=args.mode,
        )

    elapsed = time.time() - start
    after_io = read_proc_io()

    write_io_diff("case 4 gpu pipeline : fused aggregate by latitude", before_io, after_io, elapsed)

    print("\n[Row Count]")
    print(result["row_count"])
    print("[number of input file]: {}".format(result["input_file_count"]))
    print("[payload columns]: {}".format(",".join(payload_columns)))
    print("[query time]: {:.6f}s".format(float(result["query_time"])))
    print("[wrapper time]: {:.6f}s".format(elapsed))


if __name__ == "__main__":
    main()
