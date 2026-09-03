#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import glob
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import duckdb


EXCLUDED_AUTO_COLUMNS = set(["grid", "time", "levs"])
io_log = "io_duckdb_multitenant.txt"


def quote_ident(name):
    return '"' + name.replace('"', '""') + '"'


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
    parser.add_argument("--vars", default="qicps")

    parser.add_argument("--join-key", default="grid")
    parser.add_argument("--group-column", default="lats")
    parser.add_argument("--dimension-file", default="grid.parquet")

    # Number of concurrent tenants / client threads
    parser.add_argument("--tenants", type=int, default=8)

    # DuckDB internal threads PER CONNECTION
    parser.add_argument("--threads-per-tenant", type=int, default=1)

    parser.add_argument(
        "--strategy",
        default="preagg-grid",
        choices=["preagg-grid", "join-first"],
    )

    parser.add_argument(
        "--read-mode",
        default="explicit-list",
        choices=["explicit-list", "glob"],
    )

    parser.add_argument("--parquet-default-options", action="store_true")
    parser.add_argument("--preserve-insertion-order", action="store_true")
    parser.add_argument("--print-sql", action="store_true")

    # Mirrors the GPU pipeline harness's --benchmark-expr: applies the same per-value expression
    # to each raw value before it's summed, so the two harnesses stay comparable when the workload
    # is made more compute-heavy than a plain SUM.
    parser.add_argument(
        "--benchmark-expr",
        default="sum",
        choices=[
            "sum",
            "sum-sumsq",
            "derived",
            "sum-sumsq-derived",
            "sigmoid",
            "relu",
            "tanh",
            "gelu",
            "softplus",
        ],
    )

    return parser.parse_args()


def parquet_scan_expression(use_default_options):
    if use_default_options:
        return "read_parquet(?)"

    return (
        "read_parquet(?, "
        "union_by_name=false, "
        "hive_partitioning=false, "
        "filename=false, "
        "file_row_number=false, "
        "binary_as_string=false)"
    )


def read_auto_payload_columns(con, parquet_path, use_default_options):
    rows = con.execute(
        """
        DESCRIBE SELECT *
        FROM {}
        """.format(parquet_scan_expression(use_default_options)),
        [parquet_path],
    ).fetchall()

    columns = []

    for name, typ, *_ in rows:
        if typ.upper() == "DOUBLE" and name not in EXCLUDED_AUTO_COLUMNS:
            columns.append(name)

    return columns


def resolve_payload_columns(con, args, parquet_paths):
    if args.vars.strip().lower() == "all":
        return read_auto_payload_columns(
            con,
            parquet_paths[0],
            args.parquet_default_options,
        )

    return [
        column.strip()
        for column in args.vars.split(",")
        if column.strip()
    ]


def resolve_fact_source(base_dir, parquet_paths, read_mode):
    if read_mode == "explicit-list":
        return parquet_paths

    return os.path.join(
        base_dir,
        "UP-*",
        "time-levs-grid.parquet",
    )


def benchmark_expr_sql(column_ref, benchmark_expr):
    """
    Mirrors BenchmarkAggregateValue() in duckdb_gpu_probe.cu: wraps a raw column value with the
    same per-value expression the GPU kernel applies before summing, so both harnesses aggregate
    the same numbers when the workload is made more compute-heavy than a plain SUM.
    """
    c = column_ref
    if benchmark_expr == "sum":
        return c
    if benchmark_expr == "sum-sumsq":
        return "(({c}) + ({c}) * ({c}))".format(c=c)
    if benchmark_expr == "derived":
        return "(({c}) + sqrt(abs({c})) + ln(1 + abs({c})) + exp(-abs({c})))".format(c=c)
    if benchmark_expr == "sum-sumsq-derived":
        return (
            "(({c}) + ({c}) * ({c}) + sqrt(abs({c})) + ln(1 + abs({c})) + exp(-abs({c})))"
        ).format(c=c)
    if benchmark_expr == "sigmoid":
        return "(1.0 / (1.0 + exp(-({c}))))".format(c=c)
    if benchmark_expr == "relu":
        return "greatest({c}, 0)".format(c=c)
    if benchmark_expr == "tanh":
        return "tanh({c})".format(c=c)
    if benchmark_expr == "gelu":
        return (
            "(0.5 * ({c}) * (1.0 + tanh(0.7978845608028654 * "
            "(({c}) + 0.044715 * ({c}) * ({c}) * ({c})))))"
        ).format(c=c)
    if benchmark_expr == "softplus":
        return (
            "(CASE WHEN ({c}) > 0 THEN ({c}) + ln(1 + exp(-({c}))) "
            "ELSE ln(1 + exp({c})) END)"
        ).format(c=c)
    return c


def build_join_first_sql(
    base_dir,
    parquet_paths,
    payload_columns,
    join_key,
    group_column,
    read_mode,
    use_default_options,
    benchmark_expr,
):
    select_parts = [
        "g.{group_col} AS {group_col}".format(
            group_col=quote_ident(group_column)
        )
    ]

    for column in payload_columns:
        quoted = quote_ident(column)
        alias = quote_ident("sum_" + column)

        select_parts.append(
            "SUM({}) AS {}".format(
                benchmark_expr_sql("a." + quoted, benchmark_expr),
                alias,
            )
        )

    select_parts.append("COUNT(*) AS row_count")

    fact_source = resolve_fact_source(
        base_dir,
        parquet_paths,
        read_mode,
    )

    return """
        SELECT
            {select_list}
        FROM {scan} AS a
        JOIN grid_dim AS g
        USING ({join_key})
        GROUP BY g.{group_column}
    """.format(
        select_list=",\n            ".join(select_parts),
        scan=parquet_scan_expression(use_default_options),
        join_key=quote_ident(join_key),
        group_column=quote_ident(group_column),
    ), [fact_source]


def build_preagg_grid_sql(
    base_dir,
    parquet_paths,
    payload_columns,
    join_key,
    group_column,
    read_mode,
    use_default_options,
    benchmark_expr,
):
    fact_select_parts = [
        "{join_key} AS {join_key}".format(
            join_key=quote_ident(join_key)
        )
    ]

    final_select_parts = [
        "g.{group_col} AS {group_col}".format(
            group_col=quote_ident(group_column)
        )
    ]

    for column in payload_columns:
        quoted = quote_ident(column)
        alias = quote_ident("sum_" + column)

        # The activation/benchmark expression is applied once, here, to each raw value -- the
        # outer re-aggregation below is a plain SUM of already-transformed partial sums, matching
        # the GPU kernel's semantics (transform each raw value once, then sum).
        fact_select_parts.append(
            "SUM({}) AS {}".format(
                benchmark_expr_sql(quoted, benchmark_expr),
                alias,
            )
        )

        final_select_parts.append(
            "SUM(a.{alias}) AS {alias}".format(
                alias=alias
            )
        )

    fact_select_parts.append("COUNT(*) AS row_count")
    final_select_parts.append("SUM(a.row_count) AS row_count")

    fact_source = resolve_fact_source(
        base_dir,
        parquet_paths,
        read_mode,
    )

    return """
        WITH fact_agg AS (
            SELECT
                {fact_select_list}
            FROM {scan}
            GROUP BY {join_key}
        )
        SELECT
            {final_select_list}
        FROM fact_agg AS a
        JOIN grid_dim AS g
        USING ({join_key})
        GROUP BY g.{group_column}
    """.format(
        fact_select_list=",\n                ".join(fact_select_parts),
        final_select_list=",\n            ".join(final_select_parts),
        scan=parquet_scan_expression(use_default_options),
        join_key=quote_ident(join_key),
        group_column=quote_ident(group_column),
    ), [fact_source]


def build_aggregate_sql(
    base_dir,
    parquet_paths,
    payload_columns,
    join_key,
    group_column,
    strategy,
    read_mode,
    use_default_options,
    benchmark_expr,
):
    if strategy == "preagg-grid":
        return build_preagg_grid_sql(
            base_dir,
            parquet_paths,
            payload_columns,
            join_key,
            group_column,
            read_mode,
            use_default_options,
            benchmark_expr,
        )

    return build_join_first_sql(
        base_dir,
        parquet_paths,
        payload_columns,
        join_key,
        group_column,
        read_mode,
        use_default_options,
        benchmark_expr,
    )


def run_tenant(
    tenant_id,
    args,
    parquet_paths,
    grid_path,
    payload_columns,
    start_barrier,
):
    """
    One Python thread == one tenant == one DuckDB connection.
    """

    con = duckdb.connect()

    if not args.preserve_insertion_order:
        con.execute("SET preserve_insertion_order=false")

    con.execute(
        "SET threads={}".format(args.threads_per_tenant)
    )

    # Each connection has its own temp table.
    con.execute(
        """
        CREATE TEMP TABLE grid_dim AS
        SELECT {join_key}, {group_column}
        FROM {scan}
        """.format(
            join_key=quote_ident(args.join_key),
            group_column=quote_ident(args.group_column),
            scan=parquet_scan_expression(
                args.parquet_default_options
            ),
        ),
        [grid_path],
    )

    query, params = build_aggregate_sql(
        args.base_dir,
        parquet_paths,
        payload_columns,
        args.join_key,
        args.group_column,
        args.strategy,
        args.read_mode,
        args.parquet_default_options,
        args.benchmark_expr,
    )

    # All tenants wait here.
    # Once every tenant is ready, queries start together.
    start_barrier.wait()

    start = time.perf_counter()

    rows = con.execute(
        query,
        params,
    ).fetchall()

    elapsed = time.perf_counter() - start

    row_count = sum(
        int(row[-1])
        for row in rows
    )

    con.close()

    return {
        "tenant_id": tenant_id,
        "elapsed": elapsed,
        "row_count": row_count,
        "group_count": len(rows),
    }


def main():
    args = parse_args()

    if args.tenants <= 0:
        raise SystemExit("--tenants must be positive")

    if args.threads_per_tenant <= 0:
        raise SystemExit("--threads-per-tenant must be positive")

    parquet_paths = sorted(
        glob.glob(
            os.path.join(
                args.base_dir,
                "UP-*",
                "time-levs-grid.parquet",
            )
        )
    )

    if not parquet_paths:
        raise SystemExit(
            "no time-levs-grid.parquet files found"
        )

    grid_paths = sorted(
        glob.glob(
            os.path.join(
                args.base_dir,
                "UP-*",
                args.dimension_file,
            )
        )
    )

    if not grid_paths:
        raise SystemExit(
            "no {} files found".format(
                args.dimension_file
            )
        )

    #
    # Use a temporary connection only to discover columns.
    #
    metadata_con = duckdb.connect()

    payload_columns = resolve_payload_columns(
        metadata_con,
        args,
        parquet_paths,
    )

    metadata_con.close()

    if not payload_columns:
        raise SystemExit(
            "no payload columns specified"
        )

    #
    # Barrier:
    # N tenant threads + main thread
    #
    start_barrier = threading.Barrier(
        args.tenants + 1
    )

    before_io = read_proc_io()

    results = []

    wall_start = time.perf_counter()

    with ThreadPoolExecutor(
        max_workers=args.tenants
    ) as executor:

        futures = []

        for tenant_id in range(args.tenants):
            future = executor.submit(
                run_tenant,
                tenant_id,
                args,
                parquet_paths,
                grid_paths[0],
                payload_columns,
                start_barrier,
            )

            futures.append(future)

        #
        # Main thread also waits.
        # When every tenant is initialized,
        # release them simultaneously.
        #
        start_barrier.wait()

        query_start = time.perf_counter()

        for future in as_completed(futures):
            result = future.result()

            results.append(result)

            print(
                "[tenant {}] "
                "time={:.6f}s "
                "rows={} "
                "groups={}".format(
                    result["tenant_id"],
                    result["elapsed"],
                    result["row_count"],
                    result["group_count"],
                )
            )

    query_wall_elapsed = time.perf_counter() - query_start
    total_wall_elapsed = time.perf_counter() - wall_start

    after_io = read_proc_io()

    write_io_diff(
        "multitenant native duckdb aggregate",
        before_io,
        after_io,
        query_wall_elapsed,
    )

    results.sort(
        key=lambda x: x["tenant_id"]
    )

    elapsed_values = [
        r["elapsed"]
        for r in results
    ]

    avg_latency = (
        sum(elapsed_values)
        / len(elapsed_values)
    )

    min_latency = min(elapsed_values)
    max_latency = max(elapsed_values)

    throughput = (
        args.tenants / query_wall_elapsed
        if query_wall_elapsed > 0
        else 0.0
    )

    print("")
    print("========== RESULT ==========")

    print(
        "[tenants]: {}".format(
            args.tenants
        )
    )

    print(
        "[threads per tenant]: {}".format(
            args.threads_per_tenant
        )
    )

    print(
        "[total possible DuckDB workers]: {}".format(
            args.tenants
            * args.threads_per_tenant
        )
    )

    print(
        "[strategy]: {}".format(
            args.strategy
        )
    )

    print(
        "[benchmark expr]: {}".format(
            args.benchmark_expr
        )
    )

    print(
        "[variable count]: {}".format(
            len(payload_columns)
        )
    )

    print(
        "[query wall time]: {:.6f}s".format(
            query_wall_elapsed
        )
    )

    print(
        "[total wall time]: {:.6f}s".format(
            total_wall_elapsed
        )
    )

    print(
        "[average tenant latency]: {:.6f}s".format(
            avg_latency
        )
    )

    print(
        "[min tenant latency]: {:.6f}s".format(
            min_latency
        )
    )

    print(
        "[max tenant latency]: {:.6f}s".format(
            max_latency
        )
    )

    print(
        "[throughput]: {:.6f} queries/s".format(
            throughput
        )
    )


if __name__ == "__main__":
    main()