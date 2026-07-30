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
join_key = "grid"
group_column = "lats"
io_log = "io_duckdb_pipeline.txt"
lib_path = os.environ.get(
    "DUCKDB_GPU_PROBE_LIB",
    "/home/jiwan/duckdb-python-pipeline/tools/dbs_gpu_probe/libduckdb_gpu_probe.so",
)


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


before_io = read_proc_io()
start = time.time()

result = duckdb.dbs_gpu_fused_lat_pipeline(
    parquet_paths,
    payload_column=var,
    join_key=join_key,
    group_column=group_column,
    dimension_file="grid.parquet",
    lib_path=lib_path,
    mode=os.environ.get("DUCKDB_GPU_PIPELINE_MODE", "pipeline-managed"),
)

elapsed = time.time() - start
after_io = read_proc_io()

write_io_diff("case 4 gpu pipeline : Avg qicps by latitude", before_io, after_io, elapsed)

print("\n[Row Count]")
print(result["row_count"])
print("[number of input file]: {}".format(result["input_file_count"]))
print("[query time]: {:.6f}s".format(float(result["query_time"])))
print("[wrapper time]: {:.6f}s".format(elapsed))
