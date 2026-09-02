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
    parser.add_argument("--read-mode", default="per-file", choices=["per-file", "glob", "sharded"])
    parser.add_argument("--print-stage-times", action="store_true")
    parser.add_argument("--reuse-dimension-mapping", action="store_true")
    parser.add_argument(
        "--infer-grid-from-row-order",
        action="store_true",
        help="do not scan the fact grid column; infer groups from fact row order matching the dimension file",
    )
    parser.add_argument("--prefetch-files", action="store_true")
    parser.add_argument("--prefetch-method", default="both", choices=["fadvise", "readahead", "both", "io_uring", "all"])
    parser.add_argument(
        "--prefetch-threads",
        type=int,
        default=None,
        help="number of background file prefetch workers",
    )
    parser.add_argument(
        "--local-io-uring-readahead",
        action="store_true",
        help="enable io_uring read-ahead inside DuckDB local file reads",
    )
    parser.add_argument("--local-io-uring-readahead-bytes", type=int, default=None)
    parser.add_argument("--local-io-uring-readahead-depth", type=int, default=None)
    parser.add_argument("--duckdb-parquet-async-prefetch", action="store_true")
    parser.add_argument(
        "--parquet-column-chunk-prefetch",
        action="store_true",
        help="force DuckDB Parquet column-chunk prefetch for local files and overlap it with scan/decode",
    )
    parser.add_argument(
        "--parquet-prefetch-workers",
        type=int,
        default=None,
        help="number of DuckDB Parquet async prefetch workers per reader",
    )
    parser.add_argument(
        "--parquet-page-prefetch",
        action="store_true",
        help="prefetch compressed Parquet page/window ranges inside DuckDB's Parquet column reader",
    )
    parser.add_argument(
        "--parquet-page-prefetch-bytes",
        type=int,
        default=None,
        help="byte window used by --parquet-page-prefetch; defaults to 1 MiB in the C++ reader",
    )
    parser.add_argument(
        "--parquet-page-prefetch-force",
        action="store_true",
        help="disable DuckDB's row-group/column prefetch and use page/window prefetch only",
    )
    parser.add_argument(
        "--parquet-pipelined-page-read",
        action="store_true",
        help="pipeline Parquet page reads by prefetching the next page/window while the current page is decoded",
    )
    parser.add_argument(
        "--parquet-page-io-queue",
        action="store_true",
        help="queue exact compressed Parquet page payload reads inside DuckDB's column reader",
    )
    parser.add_argument(
        "--parquet-pipelined-page-read-bytes",
        type=int,
        default=None,
        help="byte window used by --parquet-pipelined-page-read; defaults to 8 MiB in the C++ reader",
    )
    parser.add_argument(
        "--fetch-raw",
        action="store_true",
        help="use DuckDB FetchRaw and consume scan vectors without the regular Fetch materialization path",
    )
    parser.add_argument(
        "--regular-fetch",
        action="store_true",
        help="force the regular DuckDB Fetch path for comparison",
    )
    parser.add_argument(
        "--parquet-direct-decode",
        action="store_true",
        help="decode Parquet DOUBLE payload columns directly into mapped GPU pipeline slot buffers",
    )
    parser.add_argument(
        "--row-order-direct-submit",
        action="store_true",
        help="with direct parquet decode, derive grid ids in the GPU kernel and submit decoded values directly",
    )
    parser.add_argument(
        "--assume-payload-all-valid",
        action="store_true",
        help="treat payload DOUBLE columns as non-null and skip validity buffer generation/copy",
    )
    parser.add_argument(
        "--reader-duckdb-threads",
        type=int,
        default=None,
        help="set DuckDB threads inside each C++ reader connection",
    )
    parser.add_argument(
        "--mode",
        default=os.environ.get("DUCKDB_GPU_PIPELINE_MODE", "pipeline-device"),
        choices=[
            "device",
            "mapped",
            "pipeline-device",
            "pipeline-device-direct",
            "pipeline-mapped",
            "pipeline-mapped-direct",
            "pipeline-managed",
            "pipeline-cpu",
            "cudf",
        ],
    )
    parser.add_argument(
        "--lib",
        default=os.environ.get(
            "DUCKDB_GPU_PROBE_LIB",
            os.path.join(os.path.dirname(__file__), "libduckdb_gpu_probe.so"),
        ),
    )
    parser.add_argument(
        "--cudf-device",
        type=int,
        default=int(os.environ.get("DUCKDB_GPU_CUDF_DEVICE", "0")),
        help="CUDA device ordinal used by --mode cudf",
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


def read_env_positive_int(name, default_value):
    value = os.environ.get(name)
    if not value:
        return default_value
    try:
        parsed = int(value)
    except ValueError:
        return default_value
    return parsed if parsed > 0 else default_value


def shard_paths(paths, shard_count):
    shard_count = max(1, min(shard_count, len(paths)))
    shards = [[] for _ in range(shard_count)]
    for idx, path in enumerate(paths):
        shards[idx % shard_count].append(path)
    return ["\n".join(shard) for shard in shards if shard]


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
    elif args.read_mode == "sharded":
        if not grid_paths:
            raise SystemExit("no {} files found".format(args.dimension_file))
        reader_threads = read_env_positive_int("DUCKDB_GPU_PIPELINE_READER_THREADS", 1)
        fact_inputs = shard_paths(parquet_paths, reader_threads)
        dimension_file = grid_paths[0]

    if args.vars and args.vars.strip().lower() == "all":
        payload_columns = read_auto_payload_columns(parquet_paths[0])
    elif args.vars:
        payload_columns = [column.strip() for column in args.vars.split(",") if column.strip()]
    else:
        payload_columns = [args.var or "qicps"]
    if not payload_columns:
        raise SystemExit("no payload columns specified")
    if args.mode == "cudf" and args.read_mode == "sharded":
        raise SystemExit("mode=cudf supports --read-mode per-file or glob")
    if args.mode == "cudf" and args.read_mode == "glob":
        fact_inputs = parquet_paths
        dimension_file = grid_paths[0]
    if args.mode == "cudf":
        if args.cudf_device < 0:
            raise SystemExit("--cudf-device must be non-negative")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.cudf_device))
        os.environ["DUCKDB_GPU_CUDF_DEVICE"] = str(args.cudf_device)
    if args.row_order_direct_submit:
        if args.mode != "pipeline-mapped-direct":
            raise SystemExit("--row-order-direct-submit requires --mode pipeline-mapped-direct")
        args.parquet_direct_decode = True
        args.assume_payload_all_valid = True
        args.infer_grid_from_row_order = True
    if len(payload_columns) > 1 and args.mode in ("pipeline-managed", "pipeline-cpu"):
        raise SystemExit(
            "{} does not support --vars yet; use pipeline-device, pipeline-device-direct, pipeline-mapped, "
            "or pipeline-mapped-direct".format(args.mode)
        )

    before_io = read_proc_io()
    start = time.time()

    if args.reuse_dimension_mapping:
        os.environ["DUCKDB_GPU_REUSE_DIMENSION_MAPPING"] = "1"
    if args.infer_grid_from_row_order:
        os.environ["DUCKDB_GPU_INFER_GRID_FROM_ROW_ORDER"] = "1"
    if args.prefetch_files:
        os.environ["DUCKDB_GPU_PREFETCH_FILES"] = "1"
        os.environ["DUCKDB_GPU_PREFETCH_METHOD"] = args.prefetch_method
    if args.prefetch_threads is not None:
        if args.prefetch_threads <= 0:
            raise SystemExit("--prefetch-threads must be positive")
        os.environ["DUCKDB_GPU_PREFETCH_THREADS"] = str(args.prefetch_threads)
    if args.local_io_uring_readahead:
        os.environ["DUCKDB_LOCAL_IO_URING_READAHEAD"] = "1"
    if args.local_io_uring_readahead_bytes is not None:
        if args.local_io_uring_readahead_bytes <= 0:
            raise SystemExit("--local-io-uring-readahead-bytes must be positive")
        os.environ["DUCKDB_LOCAL_IO_URING_READAHEAD_BYTES"] = str(args.local_io_uring_readahead_bytes)
    if args.local_io_uring_readahead_depth is not None:
        if args.local_io_uring_readahead_depth <= 0:
            raise SystemExit("--local-io-uring-readahead-depth must be positive")
        os.environ["DUCKDB_LOCAL_IO_URING_READAHEAD_DEPTH"] = str(args.local_io_uring_readahead_depth)
    if args.duckdb_parquet_async_prefetch:
        os.environ["DUCKDB_PARQUET_ASYNC_PREFETCH"] = "1"
    if args.parquet_column_chunk_prefetch:
        os.environ["DUCKDB_PARQUET_COLUMN_CHUNK_PREFETCH"] = "1"
        os.environ.setdefault("DUCKDB_PARQUET_ASYNC_PREFETCH", "1")
        os.environ.setdefault("DUCKDB_PARQUET_ASYNC_PREFETCH_WORKERS", "4")
    if args.parquet_prefetch_workers is not None:
        if args.parquet_prefetch_workers <= 0:
            raise SystemExit("--parquet-prefetch-workers must be positive")
        os.environ["DUCKDB_PARQUET_ASYNC_PREFETCH_WORKERS"] = str(args.parquet_prefetch_workers)
    if args.parquet_page_prefetch:
        os.environ["DUCKDB_PARQUET_PAGE_PREFETCH"] = "1"
        os.environ.setdefault("DUCKDB_PARQUET_ASYNC_PREFETCH", "1")
        os.environ.setdefault("DUCKDB_PARQUET_ASYNC_SINGLE_PREFETCH", "1")
    if args.parquet_page_prefetch_force:
        os.environ["DUCKDB_PARQUET_PAGE_PREFETCH"] = "1"
        os.environ["DUCKDB_PARQUET_PAGE_PREFETCH_ONLY"] = "1"
        os.environ.setdefault("DUCKDB_PARQUET_ASYNC_PREFETCH", "1")
        os.environ.setdefault("DUCKDB_PARQUET_ASYNC_SINGLE_PREFETCH", "1")
    if args.parquet_page_prefetch_bytes is not None:
        if args.parquet_page_prefetch_bytes <= 0:
            raise SystemExit("--parquet-page-prefetch-bytes must be positive")
        os.environ["DUCKDB_PARQUET_PAGE_PREFETCH_BYTES"] = str(args.parquet_page_prefetch_bytes)
    if args.parquet_pipelined_page_read:
        os.environ["DUCKDB_PARQUET_PIPELINED_PAGE_READ"] = "1"
        os.environ.setdefault("DUCKDB_PARQUET_ASYNC_PREFETCH", "1")
        os.environ.setdefault("DUCKDB_PARQUET_ASYNC_SINGLE_PREFETCH", "1")
    if args.parquet_page_io_queue:
        os.environ["DUCKDB_PARQUET_PAGE_IO_QUEUE"] = "1"
        os.environ.setdefault("DUCKDB_PARQUET_ASYNC_PREFETCH", "1")
        os.environ.setdefault("DUCKDB_PARQUET_ASYNC_SINGLE_PREFETCH", "1")
    if args.parquet_pipelined_page_read_bytes is not None:
        if args.parquet_pipelined_page_read_bytes <= 0:
            raise SystemExit("--parquet-pipelined-page-read-bytes must be positive")
        os.environ["DUCKDB_PARQUET_PIPELINED_PAGE_READ_BYTES"] = str(args.parquet_pipelined_page_read_bytes)
    if args.parquet_direct_decode:
        os.environ["DUCKDB_GPU_PARQUET_DIRECT_DECODE"] = "1"
    if args.row_order_direct_submit:
        os.environ["DUCKDB_GPU_ROW_ORDER_DIRECT_SUBMIT"] = "1"
        os.environ.setdefault("DUCKDB_GPU_PARQUET_DIRECT_DECODE", "1")
        os.environ.setdefault("DUCKDB_GPU_ASSUME_PAYLOAD_ALL_VALID", "1")
        os.environ.setdefault("DUCKDB_GPU_INFER_GRID_FROM_ROW_ORDER", "1")
    if args.assume_payload_all_valid:
        os.environ["DUCKDB_GPU_ASSUME_PAYLOAD_ALL_VALID"] = "1"
    if args.fetch_raw or (not args.regular_fetch and args.mode.startswith("pipeline-")):
        os.environ["DUCKDB_GPU_FETCH_RAW"] = "1"
    if args.reader_duckdb_threads is not None:
        if args.reader_duckdb_threads <= 0:
            raise SystemExit("--reader-duckdb-threads must be positive")
        os.environ["DUCKDB_GPU_READER_DUCKDB_THREADS"] = str(args.reader_duckdb_threads)
    elif read_env_positive_int("DUCKDB_GPU_PIPELINE_READER_THREADS", 1) > 1:
        os.environ.setdefault("DUCKDB_GPU_READER_DUCKDB_THREADS", "1")

    if args.mode == "cudf":
        result = duckdb.dbs_gpu_cudf_lat_multi(
            fact_inputs,
            payload_columns=payload_columns,
            join_key=args.join_key,
            group_column=args.group_column,
            dimension_file=dimension_file,
            read_mode=args.read_mode,
            reuse_dimension_mapping=args.reuse_dimension_mapping,
            assume_payload_all_valid=args.assume_payload_all_valid,
        )
    elif len(payload_columns) == 1:
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
    if "scan_decode_engine" in result:
        print("[scan/decode engine]: {}".format(result["scan_decode_engine"]))
    if args.mode == "cudf":
        print("[cudf device]: {}".format(args.cudf_device))
    print("[reader threads]: {}".format(os.environ.get("DUCKDB_GPU_PIPELINE_READER_THREADS", "1")))
    print("[reader duckdb threads]: {}".format(os.environ.get("DUCKDB_GPU_READER_DUCKDB_THREADS", "default")))
    if args.prefetch_files:
        print("[prefetch]: {}".format(args.prefetch_method))
        print("[prefetch threads]: {}".format(os.environ.get("DUCKDB_GPU_PREFETCH_THREADS", "1")))
    if os.environ.get("DUCKDB_LOCAL_IO_URING_READAHEAD") == "1":
        print(
            "[local io_uring readahead]: bytes={} depth={}".format(
                os.environ.get("DUCKDB_LOCAL_IO_URING_READAHEAD_BYTES", "default"),
                os.environ.get("DUCKDB_LOCAL_IO_URING_READAHEAD_DEPTH", "default"),
            )
        )
    if args.duckdb_parquet_async_prefetch:
        print("[duckdb parquet async prefetch]: on")
    if os.environ.get("DUCKDB_PARQUET_COLUMN_CHUNK_PREFETCH") == "1":
        print(
            "[duckdb parquet column chunk prefetch]: workers={}".format(
                os.environ.get("DUCKDB_PARQUET_ASYNC_PREFETCH_WORKERS", "1")
            )
        )
    if os.environ.get("DUCKDB_PARQUET_PAGE_PREFETCH") == "1":
        print("[duckdb parquet page prefetch]: bytes={}".format(os.environ.get("DUCKDB_PARQUET_PAGE_PREFETCH_BYTES", "default")))
    if os.environ.get("DUCKDB_PARQUET_PAGE_PREFETCH_ONLY") == "1":
        print("[duckdb parquet page prefetch only]: on")
    if os.environ.get("DUCKDB_PARQUET_PIPELINED_PAGE_READ") == "1":
        print(
            "[duckdb parquet pipelined page read]: bytes={}".format(
                os.environ.get("DUCKDB_PARQUET_PIPELINED_PAGE_READ_BYTES", "default")
            )
        )
    if os.environ.get("DUCKDB_PARQUET_PAGE_IO_QUEUE") == "1":
        print("[duckdb parquet page io queue]: on")
    if os.environ.get("DUCKDB_GPU_PARQUET_DIRECT_DECODE") == "1":
        print("[parquet direct decode]: on")
        if args.mode == "pipeline-mapped-direct" and args.infer_grid_from_row_order:
            print("[direct decode emit split]: on")
    if os.environ.get("DUCKDB_GPU_ROW_ORDER_DIRECT_SUBMIT") == "1":
        print("[row-order direct submit]: on")
    if os.environ.get("DUCKDB_GPU_ASSUME_PAYLOAD_ALL_VALID") == "1":
        print("[assume payload all valid]: on")
    if os.environ.get("DUCKDB_GPU_FETCH_RAW") == "1":
        print("[duckdb fetch raw]: on")
    if args.infer_grid_from_row_order:
        print("[grid inference]: row-order")
    print("[variable count]: {}".format(len(payload_columns)))
    if args.print_stage_times and "stage_times" in result:
        stage = result["stage_times"]
        if "gpu_read_time" in stage:
            print(
                "[stage work] gpu_read={:.6f}s gpu_join={:.6f}s gpu_groupby={:.6f}s gpu_merge={:.6f}s".format(
                    float(stage.get("gpu_read_time", 0.0)),
                    float(stage.get("gpu_join_time", 0.0)),
                    float(stage.get("gpu_groupby_time", 0.0)),
                    float(stage.get("gpu_merge_time", 0.0)),
                )
            )
            print("[query time]: {:.6f}s".format(float(result["query_time"])))
            print("[wrapper time]: {:.6f}s".format(elapsed))
            return
        print(
            "[stage total] read={:.6f}s prepare={:.6f}s gpu={:.6f}s merge={:.6f}s".format(
                float(stage.get("read_time", 0.0)),
                float(stage.get("prepare_time", 0.0)),
                float(stage.get("gpu_time", 0.0)),
                float(stage.get("merge_time", 0.0)),
            )
        )
        print(
            "[stage work] read_setup={:.6f}s read_fetch={:.6f}s prepare={:.6f}s gpu={:.6f}s merge={:.6f}s".format(
                float(stage.get("read_setup_time", 0.0)),
                float(stage.get("read_fetch_time", 0.0)),
                float(stage.get("prepare_work_time", 0.0)),
                float(stage.get("gpu_work_time", 0.0)),
                float(stage.get("merge_work_time", 0.0)),
            )
        )
        print(
            "[read detail] thread_max={:.6f}s connection={:.6f}s mapping_lock={:.6f}s mapping={:.6f}s "
            "query_build={:.6f}s query_submit={:.6f}s fetch={:.6f}s".format(
                float(stage.get("read_thread_max_time", 0.0)),
                float(stage.get("read_connection_time", 0.0)),
                float(stage.get("read_mapping_lock_time", 0.0)),
                float(stage.get("read_mapping_time", 0.0)),
                float(stage.get("read_query_build_time", 0.0)),
                float(stage.get("read_query_submit_time", 0.0)),
                float(stage.get("read_fetch_time", 0.0)),
            )
        )
        print(
            "[read fetch] files={} rows={} chunks={} calls={} finished_calls={} "
            "nonempty={:.6f}s finished={:.6f}s max_call={:.6f}s chunk_object={:.6f}s".format(
                int(stage.get("read_files", 0)),
                int(stage.get("read_rows", 0)),
                int(stage.get("read_chunks", 0)),
                int(stage.get("read_fetch_calls", 0)),
                int(stage.get("read_finished_fetches", 0)),
                float(stage.get("read_fetch_nonempty_time", 0.0)),
                float(stage.get("read_fetch_finished_time", 0.0)),
                float(stage.get("read_fetch_max_time", 0.0)),
                float(stage.get("read_chunk_object_time", 0.0)),
            )
        )
        if any(
            float(stage.get(key, 0.0)) > 0.0
            for key in (
                "direct_slot_start_time",
                "direct_output_chunk_time",
                "direct_finish_chunk_time",
                "direct_flush_time",
                "direct_decode_materialize_time",
            )
        ):
            print(
                "[direct prepare] slot_start={:.6f}s output_chunk={:.6f}s "
                "finish_chunk={:.6f}s flush={:.6f}s".format(
                    float(stage.get("direct_slot_start_time", 0.0)),
                    float(stage.get("direct_output_chunk_time", 0.0)),
                    float(stage.get("direct_finish_chunk_time", 0.0)),
                    float(stage.get("direct_flush_time", 0.0)),
                )
            )
            print(
                "[direct decode/materialize] {:.6f}s".format(
                    float(stage.get("direct_decode_materialize_time", 0.0))
                )
            )
        print(
            "[stage queue] read_push={:.6f}s prepare_pop={:.6f}s prepare_push={:.6f}s "
            "gpu_pop={:.6f}s gpu_push={:.6f}s merge_pop={:.6f}s".format(
                float(stage.get("read_push_time", 0.0)),
                float(stage.get("prepare_pop_time", 0.0)),
                float(stage.get("prepare_push_time", 0.0)),
                float(stage.get("gpu_pop_time", 0.0)),
                float(stage.get("gpu_push_time", 0.0)),
                float(stage.get("merge_pop_time", 0.0)),
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
        if int(stage.get("direct_decode_queue_depth", 0)) > 0:
            print("[direct decode queue] depth={}".format(int(stage.get("direct_decode_queue_depth", 0))))
        if int(stage.get("pipeline_slots", 0)) > 0:
            print("[pipeline slots] {}".format(int(stage.get("pipeline_slots", 0))))
        print(
            "[dimension mapping] reads={} reuses={}".format(
                int(stage.get("dimension_mapping_reads", 0)),
                int(stage.get("dimension_mapping_reuses", 0)),
            )
        )
        print(
            "[parquet detail] header={:.6f}s payload_read={:.6f}s decompress={:.6f}s "
            "prepare_page={:.6f}s decode={:.6f}s page_prefetch={:.6f}s".format(
                float(stage.get("parquet_page_header_time", 0.0)),
                float(stage.get("parquet_page_payload_read_time", 0.0)),
                float(stage.get("parquet_page_decompress_time", 0.0)),
                float(stage.get("parquet_page_prepare_time", 0.0)),
                float(stage.get("parquet_page_decode_time", 0.0)),
                float(stage.get("parquet_page_prefetch_time", 0.0)),
            )
        )
        print(
            "[parquet counts] pages={} payload_bytes={} decoded_rows={} decode_calls={} "
            "prefetch_ranges={} prefetch_bytes={}".format(
                int(stage.get("parquet_pages", 0)),
                int(stage.get("parquet_page_payload_bytes", 0)),
                int(stage.get("parquet_decoded_rows", 0)),
                int(stage.get("parquet_decode_calls", 0)),
                int(stage.get("parquet_prefetch_ranges", 0)),
                int(stage.get("parquet_prefetch_bytes", 0)),
            )
        )
    print("[query time]: {:.6f}s".format(float(result["query_time"])))
    print("[wrapper time]: {:.6f}s".format(elapsed))


if __name__ == "__main__":
    main()
