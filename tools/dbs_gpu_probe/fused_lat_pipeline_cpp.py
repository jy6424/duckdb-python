#!/usr/bin/env python3
import argparse
import glob
import os
import time

import duckdb


def main():
    parser = argparse.ArgumentParser(description="C++ DuckDB fused join+aggregate runner")
    parser.add_argument("base_dir")
    parser.add_argument("--var", default="qicps")
    parser.add_argument("--join-key", default="grid")
    parser.add_argument("--group-column", default="lats")
    parser.add_argument("--fact-file", default="time-levs-grid.parquet")
    parser.add_argument("--dimension-file", default="grid.parquet")
    parser.add_argument("--max-files", type=int, default=0, help="Process only the first N files when N > 0")
    parser.add_argument(
        "--mode",
        choices=["device", "mapped", "pipeline-device", "pipeline-mapped", "pipeline-managed", "pipeline-cpu"],
        default="pipeline-managed",
    )
    parser.add_argument("--print-averages", action="store_true")
    parser.add_argument(
        "--lib",
        default=os.path.join(os.path.dirname(__file__), "libduckdb_gpu_probe.so"),
        help="Path to libduckdb_gpu_probe.so",
    )
    args = parser.parse_args()

    fact_paths = sorted(glob.glob(os.path.join(args.base_dir, "UP-*", args.fact_file)))
    if args.max_files > 0:
        fact_paths = fact_paths[: args.max_files]
    if not fact_paths:
        raise RuntimeError("no input parquet files found below {}".format(args.base_dir))

    start = time.time()
    result = duckdb.dbs_gpu_fused_lat_pipeline(
        fact_paths,
        payload_column=args.var,
        join_key=args.join_key,
        group_column=args.group_column,
        dimension_file=args.dimension_file,
        lib_path=args.lib,
        mode=args.mode,
    )
    elapsed = time.time() - start

    print("\n[Row Count]")
    print(result["row_count"])

    if args.print_averages:
        print("\n[Avg {} by {} across Parquet files]".format(args.var, args.group_column))
        for group in result["groups"]:
            count = int(group["count"])
            if count == 0:
                continue
            avg = float(group["sum"]) / count
            print("({:.17g}, {:.17g})".format(float(group["group"]), avg))

    print("[number of input file]: {}".format(result["input_file_count"]))
    print("[query time]: {:.6f}s".format(float(result["query_time"])))
    print("[wrapper time]: {:.6f}s".format(elapsed))


if __name__ == "__main__":
    main()
