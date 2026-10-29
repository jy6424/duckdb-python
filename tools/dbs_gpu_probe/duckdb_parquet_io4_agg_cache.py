#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import glob
import os
import time

import duckdb


EXCLUDED_AUTO_COLUMNS = set(["grid", "time", "levs"])
io_log = "io_duckdb_agg_cache.txt"


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
    parser = argparse.ArgumentParser(description="DuckDB pre-aggregated cache for case 4 cold-cache tests")
    parser.add_argument("base_dir")
    parser.add_argument("--cache-db", default="duckdb_case4_agg_cache.duckdb")
    parser.add_argument("--table", default="case4_agg_cache")
    parser.add_argument("--mode", default="query", choices=["build", "query", "build-query"])
    parser.add_argument("--vars", default="qicps")
    parser.add_argument("--join-key", default="grid")
    parser.add_argument("--group-column", default="lats")
    parser.add_argument("--dimension-file", default="grid.parquet")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--print-sql", action="store_true")
    return parser.parse_args()


def parquet_scan_expression():
    return (
        "read_parquet(?, "
        "union_by_name=false, "
        "hive_partitioning=false, "
        "filename=false, "
        "file_row_number=false, "
        "binary_as_string=false)"
    )


def read_auto_payload_columns(con, parquet_path):
    rows = con.execute(
        """
        DESCRIBE SELECT *
        FROM {}
        """.format(parquet_scan_expression()),
        [parquet_path],
    ).fetchall()
    columns = []
    for name, typ, *_ in rows:
        if typ.upper() == "DOUBLE" and name not in EXCLUDED_AUTO_COLUMNS:
            columns.append(name)
    return columns


def resolve_payload_columns(con, args, parquet_paths):
    if args.vars.strip().lower() == "all":
        return read_auto_payload_columns(con, parquet_paths[0])
    return [column.strip() for column in args.vars.split(",") if column.strip()]


def table_exists(con, table_name):
    return con.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = ?
        """,
        [table_name],
    ).fetchone()[0] > 0


def build_cache_sql(table_name, payload_columns, join_key, group_column):
    fact_select_parts = ["{join_key} AS {join_key}".format(join_key=quote_ident(join_key))]
    final_select_parts = ["g.{group_col} AS {group_col}".format(group_col=quote_ident(group_column))]

    for column in payload_columns:
        quoted = quote_ident(column)
        alias = quote_ident("sum_" + column)
        fact_select_parts.append("SUM({}) AS {}".format(quoted, alias))
        final_select_parts.append("SUM(a.{alias}) AS {alias}".format(alias=alias))

    fact_select_parts.append("COUNT(*) AS row_count")
    final_select_parts.append("SUM(a.row_count) AS row_count")

    return """
        CREATE TABLE {table_name} AS
        WITH grid_dim AS (
            SELECT {join_key}, {group_column}
            FROM {grid_scan}
        ),
        fact_agg AS (
            SELECT
                {fact_select_list}
            FROM {fact_scan}
            GROUP BY {join_key}
        )
        SELECT
            {final_select_list}
        FROM fact_agg AS a
        JOIN grid_dim AS g
        USING ({join_key})
        GROUP BY g.{group_column}
        ORDER BY g.{group_column}
    """.format(
        table_name=quote_ident(table_name),
        join_key=quote_ident(join_key),
        group_column=quote_ident(group_column),
        grid_scan=parquet_scan_expression(),
        fact_scan=parquet_scan_expression(),
        fact_select_list=",\n                ".join(fact_select_parts),
        final_select_list=",\n            ".join(final_select_parts),
    )


def build_metadata(con, table_name, base_dir, payload_columns, parquet_paths):
    meta_table = table_name + "__meta"
    con.execute("DROP TABLE IF EXISTS {}".format(quote_ident(meta_table)))
    con.execute(
        """
        CREATE TABLE {meta_table} AS
        SELECT
            ? AS base_dir,
            ? AS variable_csv,
            ? AS file_count,
            current_timestamp AS built_at
        """.format(meta_table=quote_ident(meta_table)),
        [base_dir, ",".join(payload_columns), len(parquet_paths)],
    )


def build_cache(con, args, parquet_paths, grid_path, payload_columns):
    if table_exists(con, args.table):
        if not args.force:
            raise SystemExit("cache table already exists; pass --force to rebuild")
        con.execute("DROP TABLE IF EXISTS {}".format(quote_ident(args.table)))

    sql = build_cache_sql(args.table, payload_columns, args.join_key, args.group_column)
    if args.print_sql:
        print(sql)
    con.execute(sql, [grid_path, parquet_paths])
    build_metadata(con, args.table, args.base_dir, payload_columns, parquet_paths)
    con.execute("CHECKPOINT")


def query_cache(con, args):
    if not table_exists(con, args.table):
        raise SystemExit("cache table does not exist; run --mode build first")
    rows = con.execute("SELECT * FROM {} ORDER BY 1".format(quote_ident(args.table))).fetchall()
    row_count = sum(int(row[-1]) for row in rows)
    return rows, row_count


def main():
    args = parse_args()
    parquet_paths = sorted(glob.glob(os.path.join(args.base_dir, "UP-*", "time-levs-grid.parquet")))
    if not parquet_paths:
        raise SystemExit("no time-levs-grid.parquet files found")
    grid_paths = sorted(glob.glob(os.path.join(args.base_dir, "UP-*", args.dimension_file)))
    if not grid_paths:
        raise SystemExit("no {} files found".format(args.dimension_file))

    con = duckdb.connect(args.cache_db)
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
    if args.mode in ("build", "build-query"):
        build_cache(con, args, parquet_paths, grid_paths[0], payload_columns)
    rows = []
    row_count = 0
    if args.mode in ("query", "build-query"):
        rows, row_count = query_cache(con, args)
    elapsed = time.time() - start
    after_io = read_proc_io()

    write_io_diff("case 4 duckdb aggregate cache {}".format(args.mode), before_io, after_io, elapsed)

    print("[duckdb threads]: {}".format(threads))
    print("[mode]: {}".format(args.mode))
    print("[cache db]: {}".format(args.cache_db))
    print("[cache table]: {}".format(args.table))
    print("[variable count]: {}".format(len(payload_columns)))
    if args.mode in ("query", "build-query"):
        print("\n[Row Count]")
        print(row_count)
        print("[group count]: {}".format(len(rows)))
    print("[query time]: {:.6f}s".format(elapsed))
    con.close()


if __name__ == "__main__":
    main()
