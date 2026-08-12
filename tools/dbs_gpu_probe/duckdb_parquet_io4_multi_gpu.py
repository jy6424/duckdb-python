#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import glob
import os
import time

import duckdb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir")
    parser.add_argument("--vars", default="qicps")
    parser.add_argument("--join-key", default="grid")
    parser.add_argument("--group-column", default="lats")
    parser.add_argument("--dimension-file", default="grid.parquet")
    parser.add_argument("--mode", default=os.environ.get("DUCKDB_GPU_MULTI_MODE", "device"), choices=["device", "mapped"])
    parser.add_argument(
        "--lib",
        default=os.environ.get(
            "DUCKDB_GPU_PROBE_LIB",
            "/home/jiwan/duckdb-python/tools/dbs_gpu_probe/libduckdb_gpu_probe.so",
        ),
    )
    parser.add_argument("--print-results", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    payload_columns = [column.strip() for column in args.vars.split(",") if column.strip()]
    parquet_paths = sorted(glob.glob(os.path.join(args.base_dir, "UP-*", "time-levs-grid.parquet")))

    start = time.time()
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

    print("\n[Row Count]")
    print(result["row_count"])
    print("[number of input file]: {}".format(result["input_file_count"]))
    print("[payload columns]: {}".format(",".join(result["payload_columns"])))
    print("[query time]: {:.6f}s".format(float(result["query_time"])))
    print("[wrapper time]: {:.6f}s".format(elapsed))

    if args.print_results:
        print("\n[Results]")
        for row in result["groups"]:
            values = ["group={:.17g}".format(float(row["group"])), "row_count={}".format(int(row["row_count"]))]
            for column in payload_columns:
                values.append("sum_{}={:.17g}".format(column, float(row["sum_" + column])))
                values.append("count_{}={}".format(column, int(row["count_" + column])))
                avg = row["avg_" + column]
                if avg is None:
                    values.append("avg_{}=NULL".format(column))
                else:
                    values.append("avg_{}={:.17g}".format(column, float(avg)))
            print(", ".join(values))


if __name__ == "__main__":
    main()
