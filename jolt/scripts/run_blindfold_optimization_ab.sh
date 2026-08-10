#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

output_dir="${OUTPUT_DIR:-benchmark-runs/stage19-blindfold-opt-compute}"
log_dir="${LOG_DIR:-logs/stage19-blindfold-opt-compute}"
workload="${WORKLOAD:-fibonacci}"
scale="${SCALE:-32}"
block_size="${BLOCK_SIZE:-256}"
measurement_runs="${MEASUREMENT_RUNS:-3}"
warmup_runs="${WARMUP_RUNS:-1}"
name_prefix="${NAME_PREFIX:-fibonacci-32}"
baseline_binary="${BASELINE_BINARY:-target/release/examples/jolt_nova_stage19_benchmark-baseline}"
optimized_binary="${OPTIMIZED_BINARY:-target/release/examples/jolt_nova_stage19_benchmark-final-cache}"
mkdir -p "$output_dir" "$log_dir"

hostname
lscpu | grep -E 'CPU\(s\)|Core|Socket|Thread|Model name|NUMA'

run_case() {
  local name="$1"
  local binary="$2"
  shift 2

  /usr/bin/time -v env "$@" "$binary" \
    --workload "$workload" \
    --scale "$scale" \
    --block-sizes "$block_size" \
    --measurement-runs "$measurement_runs" \
    --warmup-runs "$warmup_runs" \
    --memory-sample-interval-ms 10 \
    --guest-target target-stage-19-guest \
    --output "$output_dir/$name.json" \
    2>&1 | tee "$log_dir/$name.log"
}

run_case \
  "$name_prefix-baseline-warm-r$measurement_runs" \
  "$baseline_binary"

run_case \
  "$name_prefix-final-cache-warm-r$measurement_runs" \
  "$optimized_binary" \
  JOLT_NOVA_PROFILE_SETUP=1
