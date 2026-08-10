#!/usr/bin/env python3
"""Strictly compare a baseline and optimized Stage-19 BlindFold artifact."""

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("optimized", type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def profile_path(artifact_path: Path) -> Path:
    return artifact_path.with_suffix(".blindfold.jsonl")


def load_profiles(path: Path) -> Dict[Tuple[int, int], Dict[str, Dict[str, Any]]]:
    rows = {}  # type: Dict[Tuple[int, int], Dict[str, Dict[str, Any]]]
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["run_index"]), int(row["block_target_size"]))
            phases = {phase["phase"]: phase for phase in row["phases"]}
            if key in rows:
                raise ValueError(f"{path}:{line_number}: duplicate profile key {key}")
            rows[key] = phases
    return rows


def validate_artifact(path: Path, artifact: Dict[str, Any]) -> None:
    samples = artifact.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{path}: no benchmark samples")
    for sample in samples:
        for field in (
            "receipt_is_production_zk",
            "native_jolt_verified",
            "streaming_proof_verified",
            "complete_zk_verified",
        ):
            if sample.get(field) is not True:
                raise ValueError(f"{path}: sample {sample.get('run_index')} failed {field}")
        if int(sample["max_resident_trace_blocks"]) > 2:
            raise ValueError(f"{path}: resident trace block bound exceeded")


def sample_values(
    artifact: Dict[str, Any],
    profiles: Dict[Tuple[int, int], Dict[str, Dict[str, Any]]],
) -> Dict[str, List[float]]:
    result = {  # type: Dict[str, List[float]]
        "blindfold_seconds": [],
        "total_seconds": [],
        "recursive_peak_mib": [],
        "recursive_delta_mib": [],
        "nova_setup_seconds": [],
        "spartan_prove_seconds": [],
    }
    for sample in artifact["samples"]:
        key = (int(sample["run_index"]), int(sample["block_target_size"]))
        phases = profiles.get(key)
        if phases is None:
            raise ValueError(f"missing BlindFold profile for sample {key}")
        result["blindfold_seconds"].append(
            float(sample["timings_micros"]["recursive_blindfold_prove"]) / 1_000_000
        )
        result["total_seconds"].append(
            float(sample["timings_micros"]["measured_total"]) / 1_000_000
        )
        result["recursive_peak_mib"].append(
            float(sample["memory_bytes"]["recursive_peak"]) / (1024 * 1024)
        )
        result["recursive_delta_mib"].append(
            float(sample["memory_bytes"]["recursive_peak_delta"]) / (1024 * 1024)
        )
        result["nova_setup_seconds"].append(
            float(phases["nova-setup"]["elapsed_micros"]) / 1_000_000
        )
        result["spartan_prove_seconds"].append(
            float(phases["spartan-prove"]["elapsed_micros"]) / 1_000_000
        )
    return result


def summary(values: List[float]) -> Dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def improvement(baseline: float, optimized: float) -> float:
    return (baseline - optimized) / baseline * 100.0


def main() -> None:
    args = parse_args()
    baseline_artifact = load_json(args.baseline)
    optimized_artifact = load_json(args.optimized)
    validate_artifact(args.baseline, baseline_artifact)
    validate_artifact(args.optimized, optimized_artifact)

    identity_fields = ("workload", "measurement_runs")
    for field in identity_fields:
        if baseline_artifact.get(field) != optimized_artifact.get(field):
            raise ValueError(f"artifacts differ in {field}")
    baseline_blocks = {sample["block_target_size"] for sample in baseline_artifact["samples"]}
    optimized_blocks = {sample["block_target_size"] for sample in optimized_artifact["samples"]}
    if baseline_blocks != optimized_blocks or len(baseline_blocks) != 1:
        raise ValueError("comparison requires the same single block size")

    baseline_values = sample_values(
        baseline_artifact, load_profiles(profile_path(args.baseline))
    )
    optimized_values = sample_values(
        optimized_artifact, load_profiles(profile_path(args.optimized))
    )
    metrics = {}  # type: Dict[str, Any]
    for name in baseline_values:
        baseline_summary = summary(baseline_values[name])
        optimized_summary = summary(optimized_values[name])
        metrics[name] = {
            "baseline": baseline_summary,
            "optimized": optimized_summary,
            "improvement_percent": improvement(
                baseline_summary["mean"], optimized_summary["mean"]
            ),
        }

    sample = baseline_artifact["samples"][0]
    result = {
        "workload": baseline_artifact["workload"],
        "block_target_size": next(iter(baseline_blocks)),
        "active_cycles": sample["active_cycles"],
        "measurement_runs": len(baseline_artifact["samples"]),
        "strict_validation_passed": True,
        "metrics": metrics,
    }

    labels = {
        "blindfold_seconds": "BlindFold (s)",
        "total_seconds": "端到端总时间 (s)",
        "nova_setup_seconds": "Nova setup (s)",
        "spartan_prove_seconds": "Spartan prove (s)",
        "recursive_peak_mib": "递归绝对峰值 RSS (MiB)",
        "recursive_delta_mib": "递归 RSS 增量 (MiB)",
    }
    order = (
        "blindfold_seconds",
        "total_seconds",
        "nova_setup_seconds",
        "spartan_prove_seconds",
        "recursive_peak_mib",
        "recursive_delta_mib",
    )
    lines = [
        "# BlindFold optimization A/B 报告",
        "",
        f"- workload：`{result['workload']}`",
        f"- block size：{result['block_target_size']}",
        f"- active cycles：{result['active_cycles']}",
        f"- measurement runs：{result['measurement_runs']}（两组均先 warmup 1 次）",
        "- 严格校验：通过（production ZK、native、streaming、完整递归证明、resident blocks ≤ 2）",
        "",
        "| 指标 | baseline 均值 ± σ | optimized 均值 ± σ | 改善 |",
        "|---|---:|---:|---:|",
    ]
    for name in order:
        metric = metrics[name]
        base = metric["baseline"]
        opt = metric["optimized"]
        lines.append(
            f"| {labels[name]} | {base['mean']:.3f} ± {base['sample_stddev']:.3f} | "
            f"{opt['mean']:.3f} ± {opt['sample_stddev']:.3f} | "
            f"{metric['improvement_percent']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "正值表示耗时或内存下降。该优化只改善同一进程中首次证明之后的热路径；冷启动仍需生成 generator key。",
            "",
        ]
    )

    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text("\n".join(lines), encoding="utf-8")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(
        "validated A/B artifacts; "
        f"BlindFold improvement={metrics['blindfold_seconds']['improvement_percent']:.2f}%"
    )


if __name__ == "__main__":
    main()
