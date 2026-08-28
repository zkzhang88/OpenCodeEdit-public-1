#!/usr/bin/env python3
"""Augment filtered OCEData to a deterministic, balanced dataset."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "OCEData"
GENERATION_DATA_DIR = REPO_ROOT / "generation" / "data"

DEFAULT_BASE = DATA_DIR / "ocedata_quality_semantic_filtered.jsonl"
DEFAULT_CANDIDATES = (
    GENERATION_DATA_DIR / "ocedata_mix_all_descriptive.jsonl",
    GENERATION_DATA_DIR / "ocedata_mix_all_lazy.jsonl",
)
DEFAULT_OUTPUT = DATA_DIR / "ocedata_quality_semantic_filtered_augmented_60k.jsonl"
DEFAULT_SUMMARY = DATA_DIR / "ocedata_quality_semantic_filtered_augmented_60k_summary.yaml"

INSTR_TYPES = (
    "ds_descriptive",
    "qwen3_descriptive",
    "ds_lazy",
    "qwen3_lazy",
)
INSTRUCTION_STYLES = ("descriptive", "lazy")
OUTPUT_FIELDS = (
    "commit",
    "code_before_purify",
    "code_after_purify",
    "instruct_purify",
    "instr_type",
)
TARGET_PER_TYPE = 15_000
SELECTION_SEED = 42


def _records(path: Path) -> Iterator[tuple[int, bytes, dict[str, Any]]]:
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL line")
            try:
                record = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be a JSON object")
            yield line_number, raw_line, record


def _validate_record(
    record: dict[str, Any], path: Path, line_number: int, *, exact_schema: bool
) -> None:
    missing = [field for field in OUTPUT_FIELDS if field not in record]
    if missing:
        raise ValueError(f"{path}:{line_number}: missing required field {missing[0]!r}")
    for field in OUTPUT_FIELDS:
        if not isinstance(record[field], str):
            raise ValueError(f"{path}:{line_number}: {field} must be a string")
    if not record["commit"]:
        raise ValueError(f"{path}:{line_number}: commit must not be empty")
    if record["instr_type"] not in INSTR_TYPES:
        raise ValueError(
            f"{path}:{line_number}: unsupported instr_type {record['instr_type']!r}"
        )
    if exact_schema and set(record) != set(OUTPUT_FIELDS):
        raise ValueError(
            f"{path}:{line_number}: base record fields do not match output schema"
        )


def _output_record(record: dict[str, Any]) -> dict[str, str]:
    return {field: record[field] for field in OUTPUT_FIELDS}


def _canonical_json(record: dict[str, str]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _selection_rank(record: dict[str, str], seed: int) -> str:
    material = (
        f"{seed}\0{record['commit']}\0{record['instr_type']}\0{_canonical_json(record)}"
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _instruction_style(instr_type: str) -> str:
    return "descriptive" if instr_type.endswith("_descriptive") else "lazy"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary(path: Path, mode: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    options: dict[str, Any] = {
        "mode": mode,
        "dir": path.parent,
        "prefix": f".{path.name}.",
        "suffix": ".tmp",
        "delete": False,
    }
    if "b" not in mode:
        options["encoding"] = "utf-8"
    handle = tempfile.NamedTemporaryFile(**options)
    return handle, Path(handle.name)


def _flush_close(handle: Any) -> None:
    handle.flush()
    os.fsync(handle.fileno())
    handle.close()


def augment_dataset(
    base_path: Path,
    candidate_paths: list[Path],
    output_path: Path,
    summary_path: Path,
    *,
    target_per_type: int = TARGET_PER_TYPE,
    target_per_style: int | None = None,
    selection_seed: int = SELECTION_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create the balanced output and return its audit summary."""
    base_path = Path(base_path)
    candidate_paths = [Path(path) for path in candidate_paths]
    output_path = Path(output_path)
    summary_path = Path(summary_path)

    if target_per_style is None:
        if target_per_type <= 0:
            raise ValueError("target_per_type must be a positive integer")
        balance_by = "instr_type"
        balance_groups = INSTR_TYPES
        target_per_group = target_per_type
    else:
        if target_per_style <= 0:
            raise ValueError("target_per_style must be a positive integer")
        balance_by = "instruction_style"
        balance_groups = INSTRUCTION_STYLES
        target_per_group = target_per_style
    inputs = {base_path.resolve(), *(path.resolve() for path in candidate_paths)}
    if output_path.resolve() in inputs or summary_path.resolve() in inputs:
        raise ValueError("output and summary paths must be distinct from input paths")
    if output_path.resolve() == summary_path.resolve():
        raise ValueError("output and summary paths must be distinct")
    if not overwrite:
        existing = [path for path in (output_path, summary_path) if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite existing file: {existing[0]}")

    base_commits: set[str] = set()
    base_counts: Counter[str] = Counter()
    base_total = 0
    for line_number, _raw_line, record in _records(base_path):
        _validate_record(record, base_path, line_number, exact_schema=True)
        base_total += 1
        base_counts[record["instr_type"]] += 1
        base_commits.add(record["commit"])

    base_group_counts: Counter[str] = Counter()
    for instr_type, count in base_counts.items():
        group = instr_type if balance_by == "instr_type" else _instruction_style(instr_type)
        base_group_counts[group] += count

    additions_needed: dict[str, int] = {}
    for group in balance_groups:
        current = base_group_counts[group]
        if current > target_per_group:
            raise ValueError(
                f"{group}: base count {current} exceeds target {target_per_group}"
            )
        additions_needed[group] = target_per_group - current

    # Resolve every cross-file and cross-type seed collision globally before
    # filling quotas. The lowest stable rank wins for each composite commit.
    winner_by_commit: dict[str, tuple[str, dict[str, str], Path, int]] = {}
    candidate_counts: Counter[str] = Counter()
    excluded_existing = 0
    duplicate_candidates = 0
    for candidate_path in candidate_paths:
        for line_number, _raw_line, record in _records(candidate_path):
            _validate_record(record, candidate_path, line_number, exact_schema=False)
            instr_type = record["instr_type"]
            candidate_counts[instr_type] += 1
            commit = record["commit"]
            if commit in base_commits:
                excluded_existing += 1
                continue
            output_record = _output_record(record)
            rank = _selection_rank(output_record, selection_seed)
            current = winner_by_commit.get(commit)
            if current is not None:
                duplicate_candidates += 1
            if current is None or rank < current[0]:
                winner_by_commit[commit] = (
                    rank,
                    output_record,
                    candidate_path,
                    line_number,
                )

    available_by_type: dict[str, list[tuple[str, dict[str, str], Path, int]]] = {
        instr_type: [] for instr_type in INSTR_TYPES
    }
    for candidate in winner_by_commit.values():
        available_by_type[candidate[1]["instr_type"]].append(candidate)
    for candidates in available_by_type.values():
        candidates.sort(key=lambda item: item[0])

    available_by_group: dict[
        str, list[tuple[str, dict[str, str], Path, int]]
    ] = {group: [] for group in balance_groups}
    for candidate in winner_by_commit.values():
        instr_type = candidate[1]["instr_type"]
        group = instr_type if balance_by == "instr_type" else _instruction_style(instr_type)
        available_by_group[group].append(candidate)
    for candidates in available_by_group.values():
        candidates.sort(key=lambda item: item[0])

    selected: list[tuple[str, dict[str, str], Path, int]] = []
    for group in balance_groups:
        required = additions_needed[group]
        found = len(available_by_group[group])
        if found < required:
            raise ValueError(f"{group}: found {found}/{required} eligible candidates")
        chosen = available_by_group[group][:required]
        selected.extend(chosen)

    selected_counts: Counter[str] = Counter(
        candidate[1]["instr_type"] for candidate in selected
    )

    # A final global rank sort gives a deterministic appended section without
    # grouping all additions by model or instruction style.
    selected.sort(key=lambda item: item[0])

    output_handle, output_temporary = _temporary(output_path, "wb")
    summary_handle = None
    summary_temporary = None
    try:
        last_base_byte = b""
        with base_path.open("rb") as base_handle:
            for chunk in iter(lambda: base_handle.read(1024 * 1024), b""):
                output_handle.write(chunk)
                last_base_byte = chunk[-1:]
        if selected and last_base_byte not in {b"\n", b"\r"}:
            output_handle.write(b"\n")
        for _rank, record, _source, _line_number in selected:
            output_handle.write(
                (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
            )
        _flush_close(output_handle)

        final_counts = {
            instr_type: base_counts[instr_type] + selected_counts[instr_type]
            for instr_type in INSTR_TYPES
        }
        summary = {
            "schema_version": 1,
            "selection_seed": selection_seed,
            "seed_definition": "full composite commit field",
            "balance": {
                "by": balance_by,
                "target_per_group": target_per_group,
                "groups": list(balance_groups),
            },
            "base_file": str(base_path),
            "base_sha256": _sha256_file(base_path),
            "candidate_files": [
                {"path": str(path), "sha256": _sha256_file(path)}
                for path in candidate_paths
            ],
            "output_file": str(output_path),
            "output_sha256": _sha256_file(output_temporary),
            "counts": {
                "base_total": base_total,
                "added_total": len(selected),
                "final_total": base_total + len(selected),
                "base_by_instr_type": {
                    key: base_counts[key] for key in INSTR_TYPES
                },
                "added_by_instr_type": {
                    key: selected_counts[key] for key in INSTR_TYPES
                },
                "final_by_instr_type": final_counts,
                "final_by_model": {
                    "ds": final_counts["ds_descriptive"] + final_counts["ds_lazy"],
                    "qwen3": final_counts["qwen3_descriptive"]
                    + final_counts["qwen3_lazy"],
                },
                "final_by_instruction_style": {
                    "descriptive": final_counts["ds_descriptive"]
                    + final_counts["qwen3_descriptive"],
                    "lazy": final_counts["ds_lazy"] + final_counts["qwen3_lazy"],
                },
            },
            "candidate_audit": {
                "records_by_instr_type": {
                    key: candidate_counts[key] for key in INSTR_TYPES
                },
                "excluded_existing_commit": excluded_existing,
                "duplicate_candidate_commit_records": duplicate_candidates,
                "globally_unique_eligible_commits": len(winner_by_commit),
                "eligible_by_instr_type_after_global_deduplication": {
                    key: len(available_by_type[key]) for key in INSTR_TYPES
                },
                "eligible_by_balance_group_after_global_deduplication": {
                    key: len(available_by_group[key]) for key in balance_groups
                },
            },
        }

        summary_handle, summary_temporary = _temporary(summary_path, "w")
        yaml.safe_dump(summary, summary_handle, sort_keys=False, allow_unicode=True)
        _flush_close(summary_handle)
        os.replace(output_temporary, output_path)
        os.replace(summary_temporary, summary_path)
        return summary
    except BaseException:
        if not output_handle.closed:
            output_handle.close()
        if summary_handle is not None and not summary_handle.closed:
            summary_handle.close()
        output_temporary.unlink(missing_ok=True)
        if summary_temporary is not None:
            summary_temporary.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Augment semantic-filtered OCEData to four balanced instruction types."
    )
    parser.add_argument("--base-file", type=Path, default=DEFAULT_BASE)
    parser.add_argument(
        "--candidate-file",
        type=Path,
        action="append",
        dest="candidate_files",
        help="Candidate JSONL; repeat for multiple files (defaults to DT-before files).",
    )
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-file", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--target-per-type", type=int, default=TARGET_PER_TYPE)
    parser.add_argument("--selection-seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = augment_dataset(
        args.base_file,
        args.candidate_files or list(DEFAULT_CANDIDATES),
        args.output_file,
        args.summary_file,
        target_per_type=args.target_per_type,
        selection_seed=args.selection_seed,
        overwrite=args.overwrite,
    )
    print(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
