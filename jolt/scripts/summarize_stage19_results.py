#!/usr/bin/env python3

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple


MIB = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize Stage-19 benchmark JSON artifacts."
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path)
    return parser.parse_args()


def mean(values: List[float]) -> float:
    return statistics.mean(values)


def pstdev(values: List[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def seconds(micros: int) -> float:
    return micros / 1_000_000


def validate_sample(path: Path, sample: dict) -> List[str]:
    failures = []
    for field in (
        "native_jolt_verified",
        "streaming_proof_verified",
        "complete_zk_verified",
        "receipt_is_production_zk",
    ):
        if sample.get(field) is not True:
            failures.append(f"{path.name}: {field} is not true")
    if sample.get("max_resident_trace_blocks", math.inf) > 2:
        failures.append(f"{path.name}: max_resident_trace_blocks exceeds 2")
    return failures


def load_artifacts(input_dir: Path) -> Tuple[List[Tuple[Path, dict]], List[str]]:
    artifacts = []
    failures = []
    for path in sorted(input_dir.glob("*.json")):
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"{path.name}: cannot read JSON: {error}")
            continue
        if artifact.get("schema_version") != "jolt-nova-stage19-benchmark-v1":
            continue
        samples = artifact.get("samples")
        if not isinstance(samples, list) or not samples:
            failures.append(f"{path.name}: missing benchmark samples")
            continue
        for sample in samples:
            failures.extend(validate_sample(path, sample))
        artifacts.append((path, artifact))
    if not artifacts:
        failures.append(f"no Stage-19 artifacts found in {input_dir}")
    return artifacts, failures


def summarize(artifacts: List[Tuple[Path, dict]]) -> List[dict]:
    groups = defaultdict(list)
    metadata = {}
    for path, artifact in artifacts:
        for sample in artifact["samples"]:
            key = (path.stem, sample["workload"], sample["block_target_size"])
            groups[key].append(sample)
            metadata[key] = artifact

    rows = []
    for key in sorted(groups):
        name, workload, block_size = key
        samples = groups[key]
        artifact = metadata[key]
        timings = [sample["timings_micros"] for sample in samples]
        memory = [sample["memory_bytes"] for sample in samples]
        proof_sizes = [sample["proof_sizes"] for sample in samples]
        totals = [seconds(item["measured_total"]) for item in timings]
        rows.append(
            {
                "file": name,
                "workload": workload,
                "runs": len(samples),
                "block_size": block_size,
                "blocks": samples[0]["block_count"],
                "cycles": samples[0]["active_cycles"],
                "native_jolt_s": mean(
                    [seconds(item["native_jolt_prove"]) for item in timings]
                ),
                "streaming_nova_s": mean(
                    [seconds(item["nova_spartan_stream"]) for item in timings]
                ),
                "blindfold_s": mean(
                    [seconds(item["recursive_blindfold_prove"]) for item in timings]
                ),
                "final_verify_s": mean(
                    [seconds(item["final_end_to_end_verify"]) for item in timings]
                ),
                "total_s": mean(totals),
                "total_stddev_s": pstdev(totals),
                "cycles_per_s": mean(
                    [item["throughput_active_cycles_per_second"] for item in samples]
                ),
                "native_memory_mib": mean(
                    [item["native_jolt_peak_delta"] / MIB for item in memory]
                ),
                "recursive_memory_mib": mean(
                    [item["recursive_peak_delta"] / MIB for item in memory]
                ),
                "jolt_proof_bytes": mean(
                    [item["native_jolt_proof"] for item in proof_sizes]
                ),
                "recursive_payload_bytes": mean(
                    [item["recursive_proof_payload"] for item in proof_sizes]
                ),
                "resident_blocks": max(
                    item["max_resident_trace_blocks"] for item in samples
                ),
                "platform": "/".join(
                    str(artifact["platform"][field])
                    for field in ("os", "arch", "build_profile", "rust_toolchain")
                ),
            }
        )
    return rows


def write_markdown(
    output: Path,
    artifacts: List[Tuple[Path, dict]],
    rows: List[dict],
    failures: List[str],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    platforms = sorted({row["platform"] for row in rows})
    lines = [
        "# Jolt-Nova Stage 19 实验结果汇总",
        "",
        f"- Stage-19 JSON artifacts：{len(artifacts)}",
        f"- 汇总配置行数：{len(rows)}",
        f"- 原始测量样本数：{sum(row['runs'] for row in rows)}",
        f"- 平台：{', '.join(f'`{item}`' for item in platforms)}",
        f"- 严格校验：{'通过' if not failures else '失败'}",
        "",
        "校验条件包括 production ZK receipt、native Jolt、streaming proof、完整递归 ZK，以及 `max_resident_trace_blocks <= 2`。",
        "",
        "## 总体结果表",
        "",
        "| 文件 | workload | runs | block | blocks | cycles | native (s) | stream/Nova (s) | BlindFold (s) | final verify (s) | total ± σ (s) | cycles/s | recursive memory (MiB) | resident blocks |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {file} | {workload} | {runs} | {block_size} | {blocks} | {cycles} | "
            "{native_jolt_s:.3f} | {streaming_nova_s:.3f} | {blindfold_s:.3f} | "
            "{final_verify_s:.3f} | {total_s:.3f} ± {total_stddev_s:.3f} | "
            "{cycles_per_s:.2f} | {recursive_memory_mib:.2f} | {resident_blocks} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## 证明大小",
            "",
            "| 文件 | workload | block | Jolt proof (bytes) | recursive payload (bytes) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {file} | {workload} | {block_size} | {jolt_proof_bytes:.0f} | "
            "{recursive_payload_bytes:.0f} |".format(**row)
        )

    if failures:
        lines.extend(["", "## 校验失败", ""])
        lines.extend(f"- {failure}" for failure in failures)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(output: Path, rows: List[dict]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    artifacts, failures = load_artifacts(args.input_dir)
    rows = summarize(artifacts)
    write_markdown(args.markdown_output, artifacts, rows, failures)
    if args.csv_output and rows:
        write_csv(args.csv_output, rows)
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    print(
        f"validated {len(artifacts)} artifacts, "
        f"summarized {len(rows)} configurations"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
