#!/usr/bin/env python3

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple


MIB = 1024 * 1024
EXPECTED_PHASES = {
    "preflight-verification",
    "artifact-preparation",
    "nova-setup",
    "spartan-setup",
    "nova-initialization",
    "nova-prove-step",
    "nova-self-verification",
    "spartan-prove",
    "proof-serialization",
    "envelope-assembly",
    "end-to-end-self-verification",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize Stage-19 BlindFold phase profiles."
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: List[float]) -> float:
    return statistics.mean(values)


def pstdev(values: List[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def load_profiles(input_dir: Path) -> Tuple[List[dict], List[str]]:
    records = []
    failures = []
    for profile_path in sorted(input_dir.glob("*.blindfold.jsonl")):
        artifact_name = profile_path.name[: -len(".blindfold.jsonl")]
        artifact_path = input_dir / f"{artifact_name}.json"
        if not artifact_path.exists():
            failures.append(f"{profile_path.name}: matching JSON artifact is missing")
            continue
        artifact = load_json(artifact_path)
        samples = {
            (sample["run_index"], sample["block_target_size"]): sample
            for sample in artifact.get("samples", [])
        }
        for line_number, line in enumerate(
            profile_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                profile = json.loads(line)
            except json.JSONDecodeError as error:
                failures.append(f"{profile_path.name}:{line_number}: {error}")
                continue
            key = (profile.get("run_index"), profile.get("block_target_size"))
            sample = samples.get(key)
            if sample is None:
                failures.append(
                    f"{profile_path.name}:{line_number}: no matching benchmark sample"
                )
                continue
            phases = profile.get("phases", [])
            phase_names = [phase.get("phase") for phase in phases]
            if set(phase_names) != EXPECTED_PHASES or len(phase_names) != len(
                EXPECTED_PHASES
            ):
                failures.append(
                    f"{profile_path.name}:{line_number}: phase set is incomplete or duplicated"
                )
                continue
            observed_micros = sum(phase["elapsed_micros"] for phase in phases)
            wrapper_micros = sample["timings_micros"]["recursive_blindfold_prove"]
            if observed_micros > wrapper_micros:
                failures.append(
                    f"{profile_path.name}:{line_number}: phase time exceeds wrapper time"
                )
            for phase in phases:
                start = phase.get("memory_start")
                peak = phase.get("memory_peak")
                delta = phase.get("memory_peak_delta")
                if start is not None and peak is not None:
                    if peak < start or delta != peak - start:
                        failures.append(
                            f"{profile_path.name}:{line_number}: invalid memory accounting"
                        )
            records.append(
                {
                    "file": artifact_name,
                    "workload": sample["workload"],
                    "run_index": key[0],
                    "block_size": key[1],
                    "cycles": sample["active_cycles"],
                    "wrapper_micros": wrapper_micros,
                    "observed_micros": observed_micros,
                    "residual_micros": wrapper_micros - observed_micros,
                    "phases": phases,
                }
            )
    if not records:
        failures.append(f"no BlindFold profile records found in {input_dir}")
    return records, failures


def aggregate(records: List[dict]) -> Tuple[List[dict], List[dict]]:
    configuration_groups = defaultdict(list)
    phase_groups = defaultdict(list)
    for record in records:
        config_key = (
            record["file"],
            record["workload"],
            record["block_size"],
            record["cycles"],
        )
        configuration_groups[config_key].append(record)
        for phase in record["phases"]:
            phase_groups[config_key + (phase["phase"],)].append(phase)

    phase_rows = []
    for key in sorted(phase_groups):
        file_name, workload, block_size, cycles, phase_name = key
        phases = phase_groups[key]
        phase_rows.append(
            {
                "file": file_name,
                "workload": workload,
                "block_size": block_size,
                "cycles": cycles,
                "phase": phase_name,
                "runs": len(phases),
                "mean_seconds": mean(
                    [phase["elapsed_micros"] / 1_000_000 for phase in phases]
                ),
                "stddev_seconds": pstdev(
                    [phase["elapsed_micros"] / 1_000_000 for phase in phases]
                ),
                "mean_peak_rss_mib": mean(
                    [
                        phase["memory_peak"] / MIB
                        for phase in phases
                        if phase.get("memory_peak") is not None
                    ]
                ),
                "mean_peak_delta_mib": mean(
                    [
                        phase["memory_peak_delta"] / MIB
                        for phase in phases
                        if phase.get("memory_peak_delta") is not None
                    ]
                ),
            }
        )

    configuration_rows = []
    by_config_phase = defaultdict(list)
    for row in phase_rows:
        by_config_phase[
            (row["file"], row["workload"], row["block_size"], row["cycles"])
        ].append(row)
    for key in sorted(configuration_groups):
        records_for_config = configuration_groups[key]
        phases = by_config_phase[key]
        wrapper_seconds = mean(
            [record["wrapper_micros"] / 1_000_000 for record in records_for_config]
        )
        observed_seconds = mean(
            [record["observed_micros"] / 1_000_000 for record in records_for_config]
        )
        dominant_time = max(phases, key=lambda row: row["mean_seconds"])
        dominant_memory = max(phases, key=lambda row: row["mean_peak_rss_mib"])
        configuration_rows.append(
            {
                "file": key[0],
                "workload": key[1],
                "block_size": key[2],
                "cycles": key[3],
                "runs": len(records_for_config),
                "wrapper_seconds": wrapper_seconds,
                "observed_seconds": observed_seconds,
                "coverage_percent": 100 * observed_seconds / wrapper_seconds,
                "residual_seconds": mean(
                    [
                        record["residual_micros"] / 1_000_000
                        for record in records_for_config
                    ]
                ),
                "dominant_time_phase": dominant_time["phase"],
                "dominant_time_seconds": dominant_time["mean_seconds"],
                "dominant_time_percent": 100
                * dominant_time["mean_seconds"]
                / wrapper_seconds,
                "peak_rss_phase": dominant_memory["phase"],
                "peak_rss_mib": dominant_memory["mean_peak_rss_mib"],
            }
        )
    return configuration_rows, phase_rows


def write_markdown(
    output: Path,
    configuration_rows: List[dict],
    phase_rows: List[dict],
    failures: List[str],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage 19 BlindFold 分阶段归因",
        "",
        f"- 配置数：{len(configuration_rows)}",
        f"- 严格校验：{'通过' if not failures else '失败'}",
        "",
        "## 配置摘要",
        "",
        "| workload | block | cycles | runs | wrapper (s) | observed (s) | coverage | dominant time phase | phase time (s) | phase share | peak RSS phase | peak RSS (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|",
    ]
    for row in configuration_rows:
        lines.append(
            "| {workload} | {block_size} | {cycles} | {runs} | "
            "{wrapper_seconds:.3f} | {observed_seconds:.3f} | {coverage_percent:.2f}% | "
            "{dominant_time_phase} | {dominant_time_seconds:.3f} | "
            "{dominant_time_percent:.2f}% | {peak_rss_phase} | {peak_rss_mib:.2f} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## 子阶段明细",
            "",
            "| workload | block | phase | runs | mean ± σ (s) | mean peak RSS (MiB) | mean phase delta (MiB) |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in phase_rows:
        lines.append(
            "| {workload} | {block_size} | {phase} | {runs} | "
            "{mean_seconds:.3f} ± {stddev_seconds:.3f} | {mean_peak_rss_mib:.2f} | "
            "{mean_peak_delta_mib:.2f} |".format(**row)
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
    records, failures = load_profiles(args.input_dir)
    configuration_rows, phase_rows = aggregate(records)
    write_markdown(args.markdown_output, configuration_rows, phase_rows, failures)
    if args.csv_output and phase_rows:
        write_csv(args.csv_output, phase_rows)
    for failure in failures:
        print(f"ERROR: {failure}")
    if failures:
        return 1
    print(
        f"validated {len(records)} profile records, "
        f"summarized {len(configuration_rows)} configurations"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
