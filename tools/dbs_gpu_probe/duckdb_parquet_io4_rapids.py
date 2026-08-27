#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import glob
import math
import os
import queue
import threading
import time


EXCLUDED_AUTO_COLUMNS = set(["grid", "time", "levs"])


def new_timers():
    return {
        "dimension_read": 0.0,
        "fact_read": 0.0,
        "join": 0.0,
        "groupby_size": 0.0,
        "groupby_sum": 0.0,
        "groupby_count": 0.0,
        "count_from_rows": 0.0,
        "merge_concat": 0.0,
        "merge_groupby": 0.0,
        "row_count_sum": 0.0,
        "reader_wall": 0.0,
    }


def add_elapsed(timers, name, start):
    timers[name] += time.time() - start


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir")
    parser.add_argument("--vars", default="qicps")
    parser.add_argument("--join-key", default="grid")
    parser.add_argument("--group-column", default="lats")
    parser.add_argument("--dimension-file", default="grid.parquet")
    parser.add_argument("--read-mode", default="per-file", choices=["per-file", "glob"])
    parser.add_argument("--reuse-dimension-mapping", action="store_true")
    parser.add_argument(
        "--infer-grid-from-row-order",
        action="store_true",
        help="skip fact grid reads and join by repeating the dimension group column in fact row order",
    )
    parser.add_argument("--assume-payload-all-valid", action="store_true")
    parser.add_argument(
        "--reader-threads",
        type=int,
        default=int(
            os.environ.get(
                "CUDF_READER_THREADS",
                os.environ.get("DUCKDB_GPU_PIPELINE_READER_THREADS", "1"),
            )
        ),
        help="number of Python reader workers used for cuDF read_parquet",
    )
    parser.add_argument(
        "--file-batch-size",
        type=int,
        default=None,
        help="number of Parquet files read by each cuDF reader task; defaults to 1 for per-file and ceil(files/threads) for glob",
    )
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


def make_counts_from_row_counts(row_counts, group_column, payload_columns):
    counts = row_counts[[group_column]].copy()
    for payload_column in payload_columns:
        counts[payload_column] = row_counts["row_count"]
    return counts


def aggregate_joined(joined, group_column, payload_columns, assume_payload_all_valid, timers):
    grouped = joined.groupby(group_column)
    timer_start = time.time()
    row_counts = grouped.size().reset_index()
    row_counts = row_counts.rename(columns={row_counts.columns[-1]: "row_count"})
    cuda_synchronize()
    add_elapsed(timers, "groupby_size", timer_start)

    timer_start = time.time()
    sums = grouped[payload_columns].sum().reset_index()
    cuda_synchronize()
    add_elapsed(timers, "groupby_sum", timer_start)

    if assume_payload_all_valid:
        timer_start = time.time()
        counts = make_counts_from_row_counts(row_counts, group_column, payload_columns)
        cuda_synchronize()
        add_elapsed(timers, "count_from_rows", timer_start)
    else:
        timer_start = time.time()
        counts = grouped[payload_columns].count().reset_index()
        cuda_synchronize()
        add_elapsed(timers, "groupby_count", timer_start)
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


def make_file_batches(paths, batch_size):
    return [paths[idx:idx + batch_size] for idx in range(0, len(paths), batch_size)]


def default_file_batch_size(read_mode, reader_threads, file_count):
    if read_mode == "glob":
        return int(math.ceil(float(file_count) / float(reader_threads)))
    return 1


def attach_group_from_row_order(cudf, fact, dim, group_column):
    dim_rows = len(dim)
    fact_rows = len(fact)
    if dim_rows <= 0:
        raise ValueError("dimension file has no rows")
    if fact_rows % dim_rows != 0:
        raise ValueError(
            "cannot infer {} from row order: fact rows {} is not a multiple of dimension rows {}".format(
                group_column, fact_rows, dim_rows
            )
        )

    repeats = fact_rows // dim_rows
    group_values = dim[group_column]
    if repeats == 1:
        fact[group_column] = group_values
    else:
        try:
            import cupy

            fact[group_column] = cudf.Series(cupy.tile(group_values.values, repeats))
        except Exception:
            fact[group_column] = cudf.concat([group_values] * repeats, ignore_index=True)
    return fact


def threaded_read_aggregate(
    cudf,
    args,
    parquet_paths,
    payload_columns,
    detail_timers,
    sum_frames,
    count_frames,
    row_count_frames,
):
    reader_threads = max(1, min(args.reader_threads, len(parquet_paths)))
    file_batch_size = args.file_batch_size
    if file_batch_size is None:
        file_batch_size = default_file_batch_size(args.read_mode, reader_threads, len(parquet_paths))
    if file_batch_size <= 0:
        raise ValueError("--file-batch-size must be positive")

    fact_columns = payload_columns if args.infer_grid_from_row_order else [args.join_key] + payload_columns
    dimension_columns = [args.group_column] if args.infer_grid_from_row_order else [args.join_key, args.group_column]

    shared_dim = None
    if args.reuse_dimension_mapping or args.infer_grid_from_row_order:
        dimension_path = resolve_dimension_path(parquet_paths[0], args.dimension_file)
        timer_start = time.time()
        shared_dim = read_parquet_gpu(cudf, dimension_path, columns=dimension_columns)
        cuda_synchronize()
        add_elapsed(detail_timers, "dimension_read", timer_start)

    tasks = queue.Queue()
    for batch_id, batch_paths in enumerate(make_file_batches(parquet_paths, file_batch_size)):
        tasks.put((batch_id, batch_paths))

    results = queue.Queue(maxsize=reader_threads * 2)
    worker_wall_times = []

    def reader_worker():
        worker_start = time.time()
        try:
            while True:
                try:
                    batch_id, batch_paths = tasks.get_nowait()
                except queue.Empty:
                    break

                read_start = time.time()
                source = batch_paths[0] if len(batch_paths) == 1 else batch_paths
                fact = read_parquet_gpu(cudf, source, columns=fact_columns)
                cuda_synchronize()
                results.put(("batch", batch_id, batch_paths, fact, time.time() - read_start))
        except BaseException as exc:
            results.put(("error", exc))
        finally:
            worker_wall_times.append(time.time() - worker_start)
            results.put(("done",))

    workers = [threading.Thread(target=reader_worker) for _ in range(reader_threads)]
    for worker in workers:
        worker.start()

    done_workers = 0
    read_seconds = 0.0
    join_seconds = 0.0
    group_seconds = 0.0

    while done_workers < reader_threads:
        item = results.get()
        item_type = item[0]
        if item_type == "done":
            done_workers += 1
            continue
        if item_type == "error":
            for worker in workers:
                worker.join()
            raise item[1]

        _, _, batch_paths, fact, fact_read_seconds = item
        detail_timers["fact_read"] += fact_read_seconds

        if shared_dim is None:
            dimension_path = resolve_dimension_path(batch_paths[0], args.dimension_file)
            timer_start = time.time()
            dim = read_parquet_gpu(cudf, dimension_path, columns=dimension_columns)
            cuda_synchronize()
            add_elapsed(detail_timers, "dimension_read", timer_start)
        else:
            dim = shared_dim

        join_start = time.time()
        if args.infer_grid_from_row_order:
            joined = attach_group_from_row_order(cudf, fact, dim, args.group_column)
        else:
            joined = fact.merge(dim, on=args.join_key, how="inner")
        cuda_synchronize()
        join_seconds += time.time() - join_start
        add_elapsed(detail_timers, "join", join_start)

        group_start = time.time()
        sums, counts, row_counts = aggregate_joined(
            joined, args.group_column, payload_columns, args.assume_payload_all_valid, detail_timers
        )
        cuda_synchronize()
        group_seconds += time.time() - group_start
        append_frame(sum_frames, sums)
        append_frame(count_frames, counts)
        append_frame(row_count_frames, row_counts)

    for worker in workers:
        worker.join()

    detail_timers["reader_wall"] = max(worker_wall_times) if worker_wall_times else 0.0
    read_seconds = detail_timers["dimension_read"] + detail_timers["reader_wall"]
    return read_seconds, join_seconds, group_seconds


def combine_frames(cudf, frames, group_column, payload_columns, timers, is_row_count=False):
    if not frames:
        columns = [group_column, "row_count"] if is_row_count else [group_column] + payload_columns
        return cudf.DataFrame({column: [] for column in columns})
    timer_start = time.time()
    combined = cudf.concat(frames, ignore_index=True)
    cuda_synchronize()
    add_elapsed(timers, "merge_concat", timer_start)

    timer_start = time.time()
    if is_row_count:
        result = combined.groupby(group_column).sum().reset_index()
    else:
        result = combined.groupby(group_column)[payload_columns].sum().reset_index()
    cuda_synchronize()
    add_elapsed(timers, "merge_groupby", timer_start)
    return result


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

    detail_timers = new_timers()
    read_seconds = 0.0
    join_seconds = 0.0
    group_seconds = 0.0
    merge_seconds = 0.0

    start = time.time()
    sum_frames = []
    count_frames = []
    row_count_frames = []

    if args.reader_threads <= 0:
        raise SystemExit("--reader-threads must be positive")

    if args.reader_threads > 1:
        read_seconds, join_seconds, group_seconds = threaded_read_aggregate(
            cudf,
            args,
            parquet_paths,
            payload_columns,
            detail_timers,
            sum_frames,
            count_frames,
            row_count_frames,
        )
    elif args.read_mode == "glob":
        dimension_path = resolve_dimension_path(parquet_paths[0], args.dimension_file)
        read_start = time.time()
        timer_start = time.time()
        dimension_columns = [args.group_column] if args.infer_grid_from_row_order else [args.join_key, args.group_column]
        dim = read_parquet_gpu(cudf, dimension_path, columns=dimension_columns)
        cuda_synchronize()
        add_elapsed(detail_timers, "dimension_read", timer_start)

        timer_start = time.time()
        fact_columns = payload_columns if args.infer_grid_from_row_order else [args.join_key] + payload_columns
        fact = read_parquet_gpu(cudf, parquet_paths, columns=fact_columns)
        cuda_synchronize()
        add_elapsed(detail_timers, "fact_read", timer_start)
        read_seconds += time.time() - read_start

        if args.infer_grid_from_row_order:
            join_start = time.time()
            joined = attach_group_from_row_order(cudf, fact, dim, args.group_column)
            cuda_synchronize()
            join_seconds += time.time() - join_start
            add_elapsed(detail_timers, "join", join_start)
        else:
            join_start = time.time()
            joined = fact.merge(dim, on=args.join_key, how="inner")
            cuda_synchronize()
            join_seconds += time.time() - join_start
            add_elapsed(detail_timers, "join", join_start)

        group_start = time.time()
        sums, counts, row_counts = aggregate_joined(
            joined, args.group_column, payload_columns, args.assume_payload_all_valid, detail_timers
        )
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
                timer_start = time.time()
                dimension_columns = (
                    [args.group_column] if args.infer_grid_from_row_order else [args.join_key, args.group_column]
                )
                cached_dim = read_parquet_gpu(cudf, dimension_path, columns=dimension_columns)
                cuda_synchronize()
                add_elapsed(detail_timers, "dimension_read", timer_start)
            dim = cached_dim
            timer_start = time.time()
            fact_columns = payload_columns if args.infer_grid_from_row_order else [args.join_key] + payload_columns
            fact = read_parquet_gpu(cudf, fact_path, columns=fact_columns)
            cuda_synchronize()
            add_elapsed(detail_timers, "fact_read", timer_start)
            read_seconds += time.time() - read_start

            if args.infer_grid_from_row_order:
                join_start = time.time()
                joined = attach_group_from_row_order(cudf, fact, dim, args.group_column)
                cuda_synchronize()
                join_seconds += time.time() - join_start
                add_elapsed(detail_timers, "join", join_start)
            else:
                join_start = time.time()
                joined = fact.merge(dim, on=args.join_key, how="inner")
                cuda_synchronize()
                join_seconds += time.time() - join_start
                add_elapsed(detail_timers, "join", join_start)

            group_start = time.time()
            sums, counts, row_counts = aggregate_joined(
                joined, args.group_column, payload_columns, args.assume_payload_all_valid, detail_timers
            )
            cuda_synchronize()
            group_seconds += time.time() - group_start
            append_frame(sum_frames, sums)
            append_frame(count_frames, counts)
            append_frame(row_count_frames, row_counts)

    merge_start = time.time()
    total_sums = combine_frames(cudf, sum_frames, args.group_column, payload_columns, detail_timers)
    total_counts = combine_frames(cudf, count_frames, args.group_column, payload_columns, detail_timers)
    total_row_counts = combine_frames(cudf, row_count_frames, args.group_column, payload_columns, detail_timers, True)
    cuda_synchronize()
    timer_start = time.time()
    row_count = int(total_row_counts["row_count"].sum()) if len(total_row_counts) else 0
    cuda_synchronize()
    add_elapsed(detail_timers, "row_count_sum", timer_start)
    # Keep final aggregate columns alive until after the last GPU operation has completed.
    _ = (total_sums, total_counts)
    merge_seconds += time.time() - merge_start

    elapsed = time.time() - start
    print("\n[Row Count]")
    print(row_count)
    print("[number of input file]: {}".format(len(parquet_paths)))
    print("[read mode]: {}".format(args.read_mode))
    print("[scan/decode engine]: cudf/libcudf")
    print("[reader threads]: {}".format(max(1, min(args.reader_threads, len(parquet_paths)))))
    if args.reader_threads > 1:
        batch_size = args.file_batch_size
        if batch_size is None:
            batch_size = default_file_batch_size(args.read_mode, args.reader_threads, len(parquet_paths))
        print("[file batch size]: {}".format(batch_size))
    if args.assume_payload_all_valid:
        print("[assume payload all valid]: on")
    if args.infer_grid_from_row_order:
        print("[grid inference]: row-order")
    print("[payload columns]: {}".format(",".join(payload_columns)))
    if args.print_stage_times:
        print(
            "[stage work] gpu_read={:.6f}s gpu_join={:.6f}s gpu_groupby={:.6f}s gpu_merge={:.6f}s".format(
                read_seconds, join_seconds, group_seconds, merge_seconds
            )
        )
        print(
            "[stage detail] dimension_read={:.6f}s fact_read={:.6f}s join={:.6f}s "
            "groupby_size={:.6f}s groupby_sum={:.6f}s groupby_count={:.6f}s "
            "count_from_rows={:.6f}s merge_concat={:.6f}s merge_groupby={:.6f}s "
            "row_count_sum={:.6f}s reader_wall={:.6f}s".format(
                detail_timers["dimension_read"],
                detail_timers["fact_read"],
                detail_timers["join"],
                detail_timers["groupby_size"],
                detail_timers["groupby_sum"],
                detail_timers["groupby_count"],
                detail_timers["count_from_rows"],
                detail_timers["merge_concat"],
                detail_timers["merge_groupby"],
                detail_timers["row_count_sum"],
                detail_timers["reader_wall"],
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
