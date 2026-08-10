#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

output_dir="benchmark-runs/stage19-scale-matrix"
log_dir="logs/stage19-scale-matrix"
mkdir -p "$output_dir" "$log_dir"

run_case() {
  local workload="$1"
  local scale="$2"
  local block_sizes="$3"
  local name="$4"

  /usr/bin/time -v cargo run --locked --release \
    -p jolt-core \
    --example jolt_nova_stage19_benchmark \
    --no-default-features \
    --features "host nova zk" \
    -- \
    --workload "$workload" \
    --scale "$scale" \
    --block-sizes "$block_sizes" \
    --measurement-runs 1 \
    --warmup-runs 0 \
    --memory-sample-interval-ms 10 \
    --guest-target target-stage-19-guest \
    --output "$output_dir/$name.json" \
    2>&1 | tee "$log_dir/$name.log"
}

run_case fibonacci 32 64,256 fibonacci-32
run_case sha3-chain 10 1024,4096 sha3-chain-10
run_case sha3-chain 100 4096,16384 sha3-chain-100
run_case sha3-chain 300 16384,65536 sha3-chain-300
run_case sha3-chain 1000 65536,262144 sha3-chain-1000

python3 scripts/summarize_stage19_results.py \
  "$output_dir" \
  --markdown-output "$output_dir/stage19-summary.md" \
  --csv-output "$output_dir/stage19-summary.csv"

python3 scripts/summarize_blindfold_profiles.py \
  "$output_dir" \
  --markdown-output "$output_dir/blindfold-summary.md" \
  --csv-output "$output_dir/blindfold-phases.csv"
