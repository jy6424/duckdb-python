#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function

import glob
import os
import sys
import time

import duckdb


base_dir = sys.argv[1]
parquet_paths = sorted(glob.glob(os.path.join(base_dir, "UP-*", "time-levs-grid.parquet")))

var = "qicps"
io_log = "io_duckdb_threads1.txt"

con = duckdb.connect()
con.execute("SET threads=1")
print("[duckdb threads]: {}".format(con.execute("SELECT current_setting('threads')").fetchone()[0]))


def quote_ident(name):
    return '"' + name.replace('"', '""') + '"'


def escape_sql_string(s):
    return s.replace("'", "''")


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


var_q = quote_ident(var)

total_sum = {}
total_count = {}
total_rows = 0

before_io = read_proc_io()
start = time.time()

for parquet_path in parquet_paths:
    print("reading: {}".format(parquet_path))

    grid_path = os.path.join(os.path.dirname(parquet_path), "grid.parquet")

    rows = con.execute(
        """
        SELECT
            g.lats,
            SUM(a.{}) AS sum_{},
            COUNT(a.{}) AS count_{},
            COUNT(*) AS row_count
        FROM read_parquet('{}') AS a
        JOIN read_parquet('{}') AS g
        USING (grid)
        GROUP BY g.lats
    """.format(
            var_q,
            var,
            var_q,
            var,
            escape_sql_string(parquet_path),
            escape_sql_string(grid_path),
        )
    ).fetchall()

    for lat, s, c, rc in rows:
        lat_key = float(lat)

        if lat_key not in total_sum:
            total_sum[lat_key] = 0.0
            total_count[lat_key] = 0

        if s is not None:
            total_sum[lat_key] += float(s)
            total_count[lat_key] += int(c)

        total_rows += int(rc)

elapsed = time.time() - start
after_io = read_proc_io()

write_io_diff("case 4 cpu threads=1 : Avg qicps by latitude", before_io, after_io, elapsed)

print("\n[Row Count]")
print(total_rows)
print("[query time]: {:.6f}s".format(elapsed))

con.close()
