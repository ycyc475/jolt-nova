#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p benchmark-runs/stage19-instrumented logs

/usr/bin/time -v cargo run --locked --release \
  -p jolt-core \
  --example jolt_nova_stage19_benchmark \
  --no-default-features \
  --features "host nova zk" \
  -- \
  --workload fibonacci \
  --scale 32 \
  --block-sizes 256 \
  --measurement-runs 1 \
  --warmup-runs 0 \
  --memory-sample-interval-ms 10 \
  --guest-target target-stage-19-guest \
  --output benchmark-runs/stage19-instrumented/smoke-fibonacci-32.json
