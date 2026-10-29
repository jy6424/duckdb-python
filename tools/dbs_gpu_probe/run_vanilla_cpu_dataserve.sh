#!/usr/bin/env bash
set -euo pipefail

PARQUET_PATTERN="${PARQUET_PATTERN:-/home/jiwan/UP-*/time-levs-grid.parquet}"
VARS="${VARS:-T,u,v,qv,hgt,p}"
GRID_START="${GRID_START:-0}"
GRID_COUNT="${GRID_COUNT:-15002}"
LIMIT="${LIMIT:-10}"
ACTIVATION="${ACTIVATION:-sigmoid}"
CPU_LOG="${CPU_LOG:-vanilla_cpu_util.log}"
RUN_LOG="${RUN_LOG:-vanilla_duckdb_cpu_run.log}"

export PARQUET_PATTERN

cd "$(dirname "$0")/../.."

sync
python - <<PY
import glob
import os

paths = glob.glob(os.environ["PARQUET_PATTERN"])
print("files:", len(paths))

for path in paths:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)

print("dropped file cache hint")
PY

rm -f "$CPU_LOG" "$RUN_LOG"

python tools/dbs_gpu_probe/dataserve_multifile_duckdb_1.py \
  --parquet-pattern "$PARQUET_PATTERN" \
  --vars "$VARS" \
  --grid-start "$GRID_START" \
  --grid-count "$GRID_COUNT" \
  --limit "$LIMIT" \
  --normalize \
  --normalize-backend numpy \
  --normalize-activation "$ACTIVATION" \
  > "$RUN_LOG" 2>&1 &

BENCH_PID=$!

pidstat -u -p "$BENCH_PID" 1 > "$CPU_LOG" &
CPU_PID=$!

wait "$BENCH_PID"

kill "$CPU_PID" 2>/dev/null || true

echo "done"
echo "CPU log: $CPU_LOG"
echo "Run log: $RUN_LOG"
