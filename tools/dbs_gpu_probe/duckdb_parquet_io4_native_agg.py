#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import glob
import os
import time

import duckdb


EXCLUDED_AUTO_COLUMNS = set(["grid", "time", "levs"])
io_log = "io_duckdb_native_agg.txt"


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
            f.write("{}: {} -> {} diff={}\n".format(k, before.get(k, 0), after[k], after[k] - before.get(k, 0)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir")
    parser.add_argument("--vars", default="qicps")
    parser.add_argument("--join-key", default="grid")
    parser.add_argument("--group-column", default="lats")
    parser.add_argument("--dimension-file", default="grid.parquet")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--strategy", default="preagg-grid", choices=["preagg-grid", "join-first"])
    parser.add_argument("--read-mode", default="explicit-list", choices=["explicit-list", "glob"])
    parser.add_argument("--parquet-default-options", action="store_true")
    parser.add_argument("--preserve-insertion-order", action="store_true")
    parser.add_argument("--print-sql", action="store_true")
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
        return read_auto_payload_columns(con, parquet_paths[0], args.parquet_default_options)
    return [column.strip() for column in args.vars.split(",") if column.strip()]


def resolve_fact_source(base_dir, parquet_paths, read_mode):
    if read_mode == "explicit-list":
        return parquet_paths
    return os.path.join(base_dir, "UP-*", "time-levs-grid.parquet")


def build_join_first_sql(base_dir, parquet_paths, payload_columns, join_key, group_column, read_mode,
                         use_default_options):
    select_parts = ["g.{group_col} AS {group_col}".format(group_col=quote_ident(group_column))]
    for column in payload_columns:
        quoted = quote_ident(column)
        alias = quote_ident("sum_" + column)
        select_parts.append("SUM(a.{}) AS {}".format(quoted, alias))
    select_parts.append("COUNT(*) AS row_count")

    fact_source = resolve_fact_source(base_dir, parquet_paths, read_mode)
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


def build_preagg_grid_sql(base_dir, parquet_paths, payload_columns, join_key, group_column, read_mode,
                          use_default_options):
    fact_select_parts = ["{join_key} AS {join_key}".format(join_key=quote_ident(join_key))]
    final_select_parts = ["g.{group_col} AS {group_col}".format(group_col=quote_ident(group_column))]
    for column in payload_columns:
        quoted = quote_ident(column)
        alias = quote_ident("sum_" + column)
        fact_select_parts.append("SUM({}) AS {}".format(quoted, alias))
        final_select_parts.append("SUM(a.{alias}) AS {alias}".format(alias=alias))
    fact_select_parts.append("COUNT(*) AS row_count")
    final_select_parts.append("SUM(a.row_count) AS row_count")

    fact_source = resolve_fact_source(base_dir, parquet_paths, read_mode)
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


def build_aggregate_sql(base_dir, parquet_paths, payload_columns, join_key, group_column, strategy, read_mode,
                        use_default_options):
    if strategy == "preagg-grid":
        return build_preagg_grid_sql(base_dir, parquet_paths, payload_columns, join_key, group_column, read_mode,
                                     use_default_options)
    return build_join_first_sql(base_dir, parquet_paths, payload_columns, join_key, group_column, read_mode,
                                use_default_options)


def main():
    args = parse_args()
    parquet_paths = sorted(glob.glob(os.path.join(args.base_dir, "UP-*", "time-levs-grid.parquet")))
    if not parquet_paths:
        raise SystemExit("no time-levs-grid.parquet files found")
    grid_paths = sorted(glob.glob(os.path.join(args.base_dir, "UP-*", args.dimension_file)))
    if not grid_paths:
        raise SystemExit("no {} files found".format(args.dimension_file))

    con = duckdb.connect()
    if not args.preserve_insertion_order:
        con.execute("SET preserve_insertion_order=false")
    if args.threads is not None:
        if args.threads <= 0:
            raise SystemExit("--threads must be positive")
        con.execute("SET threads={}".format(args.threads))
    threads = con.execute("SELECT current_setting('threads')").fetchone()[0]

    payload_columns = resolve_payload_columns(con, args, parquet_paths)
    if not payload_columns:
        raise SystemExit("no payload columns specified")

    before_io = read_proc_io()
    start = time.time()

    con.execute(
        """
        CREATE TEMP TABLE grid_dim AS
        SELECT {join_key}, {group_column}
        FROM {scan}
        """.format(
            join_key=quote_ident(args.join_key),
            group_column=quote_ident(args.group_column),
            scan=parquet_scan_expression(args.parquet_default_options),
        ),
        [grid_paths[0]],
    )

    query, params = build_aggregate_sql(args.base_dir, parquet_paths, payload_columns, args.join_key, args.group_column,
                                        args.strategy, args.read_mode, args.parquet_default_options)
    if args.print_sql:
        print(query)
    rows = con.execute(query, params).fetchall()

    elapsed = time.time() - start
    after_io = read_proc_io()
    write_io_diff("case 4 native duckdb aggregate", before_io, after_io, elapsed)

    row_count = sum(int(row[-1]) for row in rows)
    print("[duckdb threads]: {}".format(threads))
    print("[strategy]: {}".format(args.strategy))
    print("[read mode]: {}".format(args.read_mode))
    print("[parquet options]: {}".format("default" if args.parquet_default_options else "scan-minimal"))
    print("[preserve insertion order]: {}".format("on" if args.preserve_insertion_order else "off"))
    print("[variable count]: {}".format(len(payload_columns)))
    print("[variables]")
    print(",".join(payload_columns))
    print("\n[Row Count]")
    print(row_count)
    print("[group count]: {}".format(len(rows)))
    print("[query time]: {:.6f}s".format(elapsed))
    con.close()


if __name__ == "__main__":
    main()
