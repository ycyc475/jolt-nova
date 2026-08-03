#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

cargo fmt --check -p jolt-core -p tracer
cargo check --locked -p jolt-core --lib --no-default-features --features "nova zk"
cargo test --locked -p jolt-core --lib --no-default-features --features "nova zk" stage19_ -- --test-threads=1
cargo test --locked -p jolt-core --example jolt_nova_stage19_benchmark --no-default-features --features "host nova zk"
cargo test --locked -p jolt-core --lib --no-default-features --features "nova zk" stage18_ -- --test-threads=1

if [[ "${JOLT_STAGE19_REAL:-0}" == "1" ]]; then
  cargo run --locked --release -p jolt-core --example jolt_nova_stage19_benchmark \
    --no-default-features --features "host nova zk" -- \
    --workload "${JOLT_STAGE19_WORKLOAD:-fibonacci}" \
    --scale "${JOLT_STAGE19_SCALE:-32}" \
    --block-sizes "${JOLT_STAGE19_BLOCK_SIZES:-64,256}" \
    --measurement-runs "${JOLT_STAGE19_RUNS:-1}" \
    --output "${JOLT_STAGE19_OUTPUT:-benchmark-runs/stage19/fibonacci-32.json}"
fi
