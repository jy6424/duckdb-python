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
    parser.add_argument("--read-mode", default="per-file", choices=["per-file", "glob"])
    parser.add_argument("--print-stage-times", action="store_true")
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
    grid_paths = sorted(glob.glob(os.path.join(args.base_dir, "UP-*", args.dimension_file)))
    if args.read_mode == "glob" and not grid_paths:
        raise SystemExit("no {} files found".format(args.dimension_file))
    fact_inputs = parquet_paths
    dimension_file = args.dimension_file
    if args.read_mode == "glob":
        fact_inputs = [os.path.join(args.base_dir, "UP-*", "time-levs-grid.parquet")]
        dimension_file = grid_paths[0]

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
            fact_inputs,
            payload_column=payload_columns[0],
            join_key=args.join_key,
            group_column=args.group_column,
            dimension_file=dimension_file,
            lib_path=args.lib,
            mode=args.mode,
        )
    else:
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

    write_io_diff("case 4 gpu pipeline : fused aggregate by latitude", before_io, after_io, elapsed)

    print("\n[Row Count]")
    print(result["row_count"])
    print("[number of input file]: {}".format(len(parquet_paths)))
    print("[read mode]: {}".format(args.read_mode))
    print("[reader threads]: {}".format(os.environ.get("DUCKDB_GPU_PIPELINE_READER_THREADS", "1")))
    print("[payload columns]: {}".format(",".join(payload_columns)))
    if args.print_stage_times and "stage_times" in result:
        stage = result["stage_times"]
        print(
            "[stage times] read={:.6f}s prepare={:.6f}s gpu={:.6f}s merge={:.6f}s".format(
                float(stage.get("read_time", 0.0)),
                float(stage.get("prepare_time", 0.0)),
                float(stage.get("gpu_time", 0.0)),
                float(stage.get("merge_time", 0.0)),
            )
        )
        print(
            "[stage counts] read_chunks={} prepared_batches={} gpu_batches={} merged_batches={}".format(
                int(stage.get("read_chunks", 0)),
                int(stage.get("prepared_batches", 0)),
                int(stage.get("gpu_batches", 0)),
                int(stage.get("merged_batches", 0)),
            )
        )
    print("[query time]: {:.6f}s".format(float(result["query_time"])))
    print("[wrapper time]: {:.6f}s".format(elapsed))


if __name__ == "__main__":
    main()
