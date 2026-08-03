#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

cargo fmt --check -p jolt-core -p tracer
cargo check --locked -p jolt-core --lib --no-default-features --features minimal
cargo check --locked -p jolt-core --lib --no-default-features --features "nova zk"
cargo test --locked -p jolt-core --lib --no-default-features --features "nova zk" stage18_ -- --nocapture --test-threads=1
cargo test --locked -p jolt-core --lib --no-default-features --features "nova zk" recursive_zk_verifier -- --test-threads=1
cargo test --locked -p jolt-core --lib --no-default-features --features "nova zk" clear_and_zk_receipts_share_the_same_recursive_execution_statement
cargo test --locked -p jolt-core --lib --no-default-features --features "host nova zk" fib_e2e_dory_zk_recursive_blindfold --no-run

if [[ "${JOLT_STAGE18_REAL_ELF:-0}" == "1" ]]; then
  cargo test --locked -p jolt-core --lib --no-default-features --features "host nova zk" fib_e2e_dory_zk_recursive_blindfold -- --nocapture --test-threads=1
fi
