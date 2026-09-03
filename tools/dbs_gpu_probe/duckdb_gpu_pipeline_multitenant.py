#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import glob
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import duckdb


EXCLUDED_AUTO_COLUMNS = set(["grid", "time", "levs"])
io_log = "io_duckdb_gpu_pipeline_multitenant.txt"


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
        f.write("[query wall time]: {:.6f}s\n".format(elapsed))
        for k in sorted(after.keys()):
            f.write(
                "{}: {} -> {} diff={}\n".format(
                    k,
                    before.get(k, 0),
                    after[k],
                    after[k] - before.get(k, 0),
                )
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir")
    parser.add_argument("--var", default=None)
    parser.add_argument("--vars", default=None)
    parser.add_argument("--join-key", default="grid")
    parser.add_argument("--group-column", default="lats")
    parser.add_argument("--dimension-file", default="grid.parquet")
    parser.add_argument("--read-mode", default="per-file", choices=["per-file", "glob", "sharded"])
    parser.add_argument("--tenants", type=int, default=2)
    parser.add_argument("--reader-threads", type=int, default=1)
    parser.add_argument("--reader-duckdb-threads", type=int, default=1)
    parser.add_argument("--pipeline-slots", type=int, default=2)
    parser.add_argument("--batch-rows", type=int, default=262144)
    parser.add_argument("--batch-chunks", type=int, default=128)
    parser.add_argument("--direct-scan-rows", type=int, default=262144)
    parser.add_argument("--reuse-dimension-mapping", action="store_true")
    parser.add_argument("--infer-grid-from-row-order", action="store_true")
    parser.add_argument("--assume-payload-all-valid", action="store_true")
    parser.add_argument("--parquet-direct-decode", action="store_true")
    parser.add_argument("--row-order-direct-submit", action="store_true")
    parser.add_argument("--row-order-stream-accumulate", action="store_true")
    parser.add_argument(
        "--materialize-results",
        action="store_true",
        help="build the full per-group Python result objects; off by default for latency/throughput tests",
    )
    parser.add_argument("--disable-column-chunk-prefetch", action="store_true")
    parser.add_argument("--disable-async-prefetch", action="store_true")
    parser.add_argument(
        "--benchmark-expr",
        default="sum",
        choices=["sum", "sum-sumsq", "derived", "sum-sumsq-derived"],
    )
    parser.add_argument(
        "--mode",
        default=os.environ.get("DUCKDB_GPU_PIPELINE_MODE", "pipeline-mapped-direct"),
        choices=[
            "device",
            "mapped",
            "pipeline-device",
            "pipeline-device-direct",
            "pipeline-mapped",
            "pipeline-mapped-direct",
            "pipeline-managed",
            "pipeline-cpu",
        ],
    )
    parser.add_argument(
        "--lib",
        default=os.environ.get(
            "DUCKDB_GPU_PROBE_LIB",
            os.path.join(os.path.dirname(__file__), "libduckdb_gpu_probe.so"),
        ),
    )
    parser.add_argument("--print-stage-times", action="store_true")
    return parser.parse_args()


def read_auto_payload_columns(parquet_path):
    con = duckdb.connect()
    try:
        rows = con.execute("""
            DESCRIBE SELECT *
            FROM read_parquet(
                ?,
                union_by_name=false,
                hive_partitioning=false,
                filename=false,
                file_row_number=false,
                binary_as_string=false
            )
        """, [parquet_path]).fetchall()
    finally:
        con.close()

    columns = []
    for name, typ, *_ in rows:
        if typ.upper() == "DOUBLE" and name not in EXCLUDED_AUTO_COLUMNS:
            columns.append(name)
    return columns


def resolve_payload_columns(args, parquet_paths):
    if args.vars and args.vars.strip().lower() == "all":
        return read_auto_payload_columns(parquet_paths[0])
    if args.vars:
        return [column.strip() for column in args.vars.split(",") if column.strip()]
    return [args.var or "qicps"]


def shard_paths(paths, shard_count):
    shard_count = max(1, min(shard_count, len(paths)))
    shards = [[] for _ in range(shard_count)]
    for idx, path in enumerate(paths):
        shards[idx % shard_count].append(path)
    return ["\n".join(shard) for shard in shards if shard]


def configure_environment(args):
    os.environ["DUCKDB_GPU_PIPELINE_READER_THREADS"] = str(args.reader_threads)
    os.environ["DUCKDB_GPU_READER_DUCKDB_THREADS"] = str(args.reader_duckdb_threads)
    os.environ["DUCKDB_GPU_PIPELINE_SLOTS"] = str(args.pipeline_slots)
    os.environ["DUCKDB_GPU_PIPELINE_BATCH_ROWS"] = str(args.batch_rows)
    os.environ["DUCKDB_GPU_PIPELINE_BATCH_CHUNKS"] = str(args.batch_chunks)
    os.environ["DUCKDB_GPU_PARQUET_DIRECT_SCAN_ROWS"] = str(args.direct_scan_rows)

    if args.reuse_dimension_mapping:
        os.environ["DUCKDB_GPU_REUSE_DIMENSION_MAPPING"] = "1"
    if args.infer_grid_from_row_order:
        os.environ["DUCKDB_GPU_INFER_GRID_FROM_ROW_ORDER"] = "1"
    if args.assume_payload_all_valid:
        os.environ["DUCKDB_GPU_ASSUME_PAYLOAD_ALL_VALID"] = "1"
    if args.parquet_direct_decode:
        os.environ["DUCKDB_GPU_PARQUET_DIRECT_DECODE"] = "1"
    if args.row_order_direct_submit:
        os.environ["DUCKDB_GPU_ROW_ORDER_DIRECT_SUBMIT"] = "1"
        os.environ.setdefault("DUCKDB_GPU_PARQUET_DIRECT_DECODE", "1")
        os.environ.setdefault("DUCKDB_GPU_PARQUET_DIRECT_DOUBLE_SCAN", "1")
        os.environ.setdefault("DUCKDB_PARQUET_DIRECT_PAGE_BUFFER", "1")
        os.environ.setdefault("DUCKDB_GPU_ASSUME_PAYLOAD_ALL_VALID", "1")
        os.environ.setdefault("DUCKDB_GPU_INFER_GRID_FROM_ROW_ORDER", "1")
        if not args.disable_column_chunk_prefetch:
            os.environ.setdefault("DUCKDB_PARQUET_COLUMN_CHUNK_PREFETCH", "1")
        if not args.disable_async_prefetch:
            os.environ.setdefault("DUCKDB_PARQUET_ASYNC_PREFETCH", "1")
            os.environ.setdefault("DUCKDB_PARQUET_ASYNC_PREFETCH_WORKERS", "4")
    if args.row_order_stream_accumulate:
        os.environ["DUCKDB_GPU_ROW_ORDER_STREAM_ACCUMULATE"] = "1"
        os.environ.setdefault("DUCKDB_GPU_ROW_ORDER_DIRECT_SUBMIT", "1")
    if args.benchmark_expr != "sum":
        os.environ["DUCKDB_GPU_COMPUTE_BENCHMARK"] = args.benchmark_expr
    if not args.materialize_results:
        os.environ["DUCKDB_GPU_SKIP_GROUP_RESULTS"] = "1"


def resolve_fact_inputs(args, parquet_paths, grid_paths):
    if args.read_mode == "glob":
        return [os.path.join(args.base_dir, "UP-*", "time-levs-grid.parquet")], grid_paths[0]
    if args.read_mode == "sharded":
        return shard_paths(parquet_paths, args.reader_threads), grid_paths[0]
    return parquet_paths, args.dimension_file


def run_tenant(tenant_id, args, fact_inputs, payload_columns, dimension_file, start_barrier):
    start_barrier.wait()
    start = time.perf_counter()
    result = duckdb.dbs_gpu_fused_lat_multi(
        fact_inputs,
        payload_columns=payload_columns,
        join_key=args.join_key,
        group_column=args.group_column,
        dimension_file=dimension_file,
        lib_path=args.lib,
        mode=args.mode,
    )
    elapsed = time.perf_counter() - start
    stage_times = result.get("stage_times", {})
    return {
        "tenant_id": tenant_id,
        "elapsed": elapsed,
        "query_time": float(result.get("query_time", elapsed)),
        "row_count": int(result["row_count"]),
        "group_count": int(result.get("group_count", len(result.get("groups", [])))),
        "read_thread_max": float(stage_times.get("read_thread_max_time", 0.0)),
        "gpu_pop": float(stage_times.get("gpu_pop_time", 0.0)),
        "parquet_decode": float(stage_times.get("parquet_page_decode_time", 0.0)),
        "parquet_direct_scan": float(stage_times.get("parquet_direct_scan_time", 0.0)),
        "pipeline_handle_reused": int(stage_times.get("pipeline_handle_reused", 0)),
        "pipeline_acquire_time": float(stage_times.get("pipeline_acquire_time", 0.0)),
        "pipeline_release_time": float(stage_times.get("pipeline_release_time", 0.0)),
        "accumulate_pop_time": float(stage_times.get("accumulate_pop_time", 0.0)),
        "accumulate_work_time": float(stage_times.get("accumulate_work_time", 0.0)),
        "accumulate_push_time": float(stage_times.get("accumulate_push_time", 0.0)),
        "stage_times_full": dict(stage_times),
    }


def main():
    args = parse_args()
    if args.tenants <= 0:
        raise SystemExit("--tenants must be positive")
    if args.reader_threads <= 0:
        raise SystemExit("--reader-threads must be positive")
    if args.reader_duckdb_threads <= 0:
        raise SystemExit("--reader-duckdb-threads must be positive")
    if args.pipeline_slots <= 0:
        raise SystemExit("--pipeline-slots must be positive")
    if args.batch_rows <= 0:
        raise SystemExit("--batch-rows must be positive")
    if args.batch_chunks <= 0:
        raise SystemExit("--batch-chunks must be positive")
    if args.direct_scan_rows <= 0:
        raise SystemExit("--direct-scan-rows must be positive")
    if args.row_order_stream_accumulate:
        args.row_order_direct_submit = True
    if args.row_order_direct_submit:
        args.parquet_direct_decode = True
        args.assume_payload_all_valid = True
        args.infer_grid_from_row_order = True

    parquet_paths = sorted(glob.glob(os.path.join(args.base_dir, "UP-*", "time-levs-grid.parquet")))
    if not parquet_paths:
        raise SystemExit("no time-levs-grid.parquet files found")
    grid_paths = sorted(glob.glob(os.path.join(args.base_dir, "UP-*", args.dimension_file)))
    if args.read_mode in ("glob", "sharded") and not grid_paths:
        raise SystemExit("no {} files found".format(args.dimension_file))

    payload_columns = resolve_payload_columns(args, parquet_paths)
    if not payload_columns:
        raise SystemExit("no payload columns specified")

    configure_environment(args)
    fact_inputs, dimension_file = resolve_fact_inputs(args, parquet_paths, grid_paths)

    start_barrier = threading.Barrier(args.tenants + 1)
    before_io = read_proc_io()
    results = []
    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.tenants) as executor:
        futures = [
            executor.submit(run_tenant, tenant_id, args, fact_inputs, payload_columns, dimension_file, start_barrier)
            for tenant_id in range(args.tenants)
        ]
        start_barrier.wait()
        query_start = time.perf_counter()
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                "[tenant {}] time={:.6f}s query={:.6f}s rows={} groups={}".format(
                    result["tenant_id"],
                    result["elapsed"],
                    result["query_time"],
                    result["row_count"],
                    result["group_count"],
                )
            )

    query_wall_elapsed = time.perf_counter() - query_start
    total_wall_elapsed = time.perf_counter() - wall_start
    after_io = read_proc_io()
    write_io_diff("multitenant gpu pipeline", before_io, after_io, query_wall_elapsed)

    results.sort(key=lambda x: x["tenant_id"])
    elapsed_values = [r["elapsed"] for r in results]
    avg_latency = sum(elapsed_values) / len(elapsed_values)
    min_latency = min(elapsed_values)
    max_latency = max(elapsed_values)
    throughput = args.tenants / query_wall_elapsed if query_wall_elapsed > 0 else 0.0

    print("")
    print("========== RESULT ==========")
    print("[tenants]: {}".format(args.tenants))
    print("[reader threads per tenant]: {}".format(args.reader_threads))
    print("[reader duckdb threads]: {}".format(args.reader_duckdb_threads))
    print("[pipeline slots per tenant]: {}".format(args.pipeline_slots))
    print("[benchmark expr]: {}".format(args.benchmark_expr))
    print("[read mode]: {}".format(args.read_mode))
    print("[mode]: {}".format(args.mode))
    print("[number of input file]: {}".format(len(parquet_paths)))
    print("[variable count]: {}".format(len(payload_columns)))
    print("[query wall time]: {:.6f}s".format(query_wall_elapsed))
    print("[total wall time]: {:.6f}s".format(total_wall_elapsed))
    print("[average tenant latency]: {:.6f}s".format(avg_latency))
    print("[min tenant latency]: {:.6f}s".format(min_latency))
    print("[max tenant latency]: {:.6f}s".format(max_latency))
    print("[throughput]: {:.6f} queries/s".format(throughput))

    if args.print_stage_times:
        for result in results:
            print(
                "[tenant {} stage] read_thread_max={:.6f}s gpu_pop={:.6f}s "
                "parquet_decode={:.6f}s parquet_direct_scan={:.6f}s "
                "pipeline_reused={} pipeline_acquire={:.6f}s pipeline_release={:.6f}s "
                "accumulate_pop={:.6f}s accumulate_work={:.6f}s accumulate_push={:.6f}s".format(
                    result["tenant_id"],
                    result["read_thread_max"],
                    result["gpu_pop"],
                    result["parquet_decode"],
                    result["parquet_direct_scan"],
                    result["pipeline_handle_reused"],
                    result["pipeline_acquire_time"],
                    result["pipeline_release_time"],
                    result["accumulate_pop_time"],
                    result["accumulate_work_time"],
                    result["accumulate_push_time"],
                )
            )


if __name__ == "__main__":
    main()
