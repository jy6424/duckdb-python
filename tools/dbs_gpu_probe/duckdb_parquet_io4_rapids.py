#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import glob
import os
import time


EXCLUDED_AUTO_COLUMNS = set(["grid", "time", "levs"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir")
    parser.add_argument("--vars", default="qicps")
    parser.add_argument("--join-key", default="grid")
    parser.add_argument("--group-column", default="lats")
    parser.add_argument("--dimension-file", default="grid.parquet")
    parser.add_argument("--read-mode", default="per-file", choices=["per-file", "glob"])
    parser.add_argument("--reuse-dimension-mapping", action="store_true")
    parser.add_argument("--print-stage-times", action="store_true")
    parser.add_argument("--print-results", action="store_true")
    return parser.parse_args()


def read_auto_payload_columns(parquet_path):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required for --vars all metadata inspection: {}".format(exc))

    schema = pq.ParquetFile(parquet_path).schema_arrow
    return [
        field.name
        for field in schema
        if field.name not in EXCLUDED_AUTO_COLUMNS and str(field.type).lower() == "double"
    ]


def resolve_payload_columns(args, parquet_paths):
    if args.vars.strip().lower() == "all":
        return read_auto_payload_columns(parquet_paths[0])
    return [column.strip() for column in args.vars.split(",") if column.strip()]


def resolve_dimension_path(fact_path, dimension_file):
    if os.path.isabs(dimension_file) or os.path.dirname(dimension_file):
        return dimension_file
    return os.path.join(os.path.dirname(fact_path), dimension_file)


def aggregate_joined(joined, group_column, payload_columns):
    grouped = joined.groupby(group_column)
    row_counts = grouped.size().reset_index(name="row_count")
    sums = grouped[payload_columns].sum().reset_index()
    counts = grouped[payload_columns].count().reset_index()
    return sums, counts, row_counts


def append_frame(frames, frame):
    if frame is not None and len(frame) > 0:
        frames.append(frame)


def cuda_synchronize():
    try:
        import cupy

        cupy.cuda.Stream.null.synchronize()
        return
    except Exception:
        pass
    try:
        from numba import cuda

        cuda.synchronize()
    except Exception:
        pass


def read_parquet_gpu(cudf, source, columns):
    return cudf.read_parquet(source, columns=columns)


def combine_frames(cudf, frames, group_column, payload_columns, is_row_count=False):
    if not frames:
        columns = [group_column, "row_count"] if is_row_count else [group_column] + payload_columns
        return cudf.DataFrame({column: [] for column in columns})
    combined = cudf.concat(frames, ignore_index=True)
    if is_row_count:
        return combined.groupby(group_column).sum().reset_index()
    return combined.groupby(group_column)[payload_columns].sum().reset_index()


def main():
    args = parse_args()
    try:
        import cudf
    except ImportError as exc:
        raise SystemExit("cudf is required for GPU-native Parquet scan: {}".format(exc))

    parquet_paths = sorted(glob.glob(os.path.join(args.base_dir, "UP-*", "time-levs-grid.parquet")))
    if not parquet_paths:
        raise SystemExit("no time-levs-grid.parquet files found")

    payload_columns = resolve_payload_columns(args, parquet_paths)
    if not payload_columns:
        raise SystemExit("no payload columns specified")

    read_seconds = 0.0
    join_seconds = 0.0
    group_seconds = 0.0
    merge_seconds = 0.0

    start = time.time()
    sum_frames = []
    count_frames = []
    row_count_frames = []

    if args.read_mode == "glob":
        dimension_path = resolve_dimension_path(parquet_paths[0], args.dimension_file)
        read_start = time.time()
        dim = read_parquet_gpu(cudf, dimension_path, columns=[args.join_key, args.group_column])
        fact = read_parquet_gpu(cudf, parquet_paths, columns=[args.join_key] + payload_columns)
        cuda_synchronize()
        read_seconds += time.time() - read_start

        join_start = time.time()
        joined = fact.merge(dim, on=args.join_key, how="inner")
        cuda_synchronize()
        join_seconds += time.time() - join_start

        group_start = time.time()
        sums, counts, row_counts = aggregate_joined(joined, args.group_column, payload_columns)
        cuda_synchronize()
        group_seconds += time.time() - group_start
        append_frame(sum_frames, sums)
        append_frame(count_frames, counts)
        append_frame(row_count_frames, row_counts)
    else:
        cached_dim = None
        for fact_path in parquet_paths:
            dimension_path = resolve_dimension_path(fact_path, args.dimension_file)
            read_start = time.time()
            if cached_dim is None or not args.reuse_dimension_mapping:
                cached_dim = read_parquet_gpu(cudf, dimension_path, columns=[args.join_key, args.group_column])
            dim = cached_dim
            fact = read_parquet_gpu(cudf, fact_path, columns=[args.join_key] + payload_columns)
            cuda_synchronize()
            read_seconds += time.time() - read_start

            join_start = time.time()
            joined = fact.merge(dim, on=args.join_key, how="inner")
            cuda_synchronize()
            join_seconds += time.time() - join_start

            group_start = time.time()
            sums, counts, row_counts = aggregate_joined(joined, args.group_column, payload_columns)
            cuda_synchronize()
            group_seconds += time.time() - group_start
            append_frame(sum_frames, sums)
            append_frame(count_frames, counts)
            append_frame(row_count_frames, row_counts)

    merge_start = time.time()
    total_sums = combine_frames(cudf, sum_frames, args.group_column, payload_columns)
    total_counts = combine_frames(cudf, count_frames, args.group_column, payload_columns)
    total_row_counts = combine_frames(cudf, row_count_frames, args.group_column, payload_columns, True)
    cuda_synchronize()
    row_count = int(total_row_counts["row_count"].sum()) if len(total_row_counts) else 0
    # Keep final aggregate columns alive until after the last GPU operation has completed.
    _ = (total_sums, total_counts)
    merge_seconds += time.time() - merge_start

    elapsed = time.time() - start
    print("\n[Row Count]")
    print(row_count)
    print("[number of input file]: {}".format(len(parquet_paths)))
    print("[read mode]: {}".format(args.read_mode))
    print("[scan/decode engine]: cudf")
    print("[payload columns]: {}".format(",".join(payload_columns)))
    if args.print_stage_times:
        print(
            "[stage work] gpu_read={:.6f}s gpu_join={:.6f}s gpu_groupby={:.6f}s gpu_merge={:.6f}s".format(
                read_seconds, join_seconds, group_seconds, merge_seconds
            )
        )
    if args.print_results:
        print("\n[Results]")
        merged = total_sums.merge(total_counts, on=args.group_column, suffixes=("_sum", "_count"))
        merged = merged.merge(total_row_counts, on=args.group_column)
        for row in merged.sort_values(args.group_column).to_pandas().itertuples(index=False):
            print(row)
    print("[query time]: {:.6f}s".format(elapsed))


if __name__ == "__main__":
    main()
