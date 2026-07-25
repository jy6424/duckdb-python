#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-${SCRIPT_DIR}/libduckdb_gpu_probe.so}"

nvcc -O3 -std=c++17 -Xcompiler=-fPIC -shared \
	"${SCRIPT_DIR}/duckdb_gpu_probe.cu" \
	-o "${OUT}"

echo "${OUT}"
