#!/usr/bin/env python3
"""Export a reproducible stratified sample of DT-filtered edit records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Dict, List, NamedTuple, Optional, Sequence


DEFAULT_INPUT_FILES = (
    Path(
        "generation/data/filtered/"
        "ocedata_mix_all_descriptive_dt_filtered.jsonl"
    ),
    Path(
        "generation/data/filtered/"
        "ocedata_mix_all_lazy_dt_filtered.jsonl"
    ),
)
DEFAULT_OUTPUT_DIR = Path(
    "generation/data/filtered/ocedata_mix_all_dt_filtered_samples"
)
DEFAULT_COUNT = 384
DEFAULT_SEED = 42
INSTR_TYPES = (
    "ds_descriptive",
    "qwen3_descriptive",
    "ds_lazy",
    "qwen3_lazy",
)
MANUAL_REVIEW_FIELD = "manual_post_edit_fulfills_instruction"


class SampleRecord(NamedTuple):
    """One validated record and its location in the source JSONL file."""

    source_file: Path
    source_line_number: int
    record: Dict[str, object]


def _required_string(
    record: object,
    field: str,
    source_file: Path,
    line_number: int,
) -> str:
    if not isinstance(record, dict):
        raise ValueError(
            f"Expected a JSON object in {source_file} at line {line_number}"
        )
    if field not in record:
        raise ValueError(
            f"{source_file}:{line_number} is missing required field '{field}'"
        )
    value = record[field]
    if not isinstance(value, str):
        raise ValueError(
            f"{source_file}:{line_number} field '{field}' must be a string"
        )
    return value


def _validate_count(count: int) -> int:
    if isinstance(count, bool) or count <= 0:
        raise ValueError("count must be a positive integer")
    if count % len(INSTR_TYPES) != 0:
        raise ValueError(
            f"count must be divisible by {len(INSTR_TYPES)} instruction types"
        )
    return count // len(INSTR_TYPES)


def sample_records(
    input_files: Sequence[Path],
    count: int = DEFAULT_COUNT,
    seed: int = DEFAULT_SEED,
) -> Dict[str, List[SampleRecord]]:
    """Select an equal-size reservoir sample for every supported type."""
    per_type = _validate_count(count)
    if not input_files:
        raise ValueError("at least one input file is required")
    for input_file in input_files:
        if not input_file.is_file():
            raise FileNotFoundError(f"Input file not found: {input_file}")

    rng = random.Random(seed)
    samples: Dict[str, List[SampleRecord]] = {
        instr_type: [] for instr_type in INSTR_TYPES
    }
    seen = {instr_type: 0 for instr_type in INSTR_TYPES}
    required_fields = (
        "code_before_purify",
        "code_after_purify",
        "instruct_purify",
        "commit",
        "instr_type",
    )

    for input_file in input_files:
        with input_file.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {input_file} at line {line_number}: {exc}"
                    ) from exc

                values = {
                    field: _required_string(
                        record, field, input_file, line_number
                    )
                    for field in required_fields
                }
                instr_type = values["instr_type"]
                if instr_type not in samples:
                    raise ValueError(
                        f"{input_file}:{line_number} has unsupported instr_type "
                        f"'{instr_type}'"
                    )

                sample = SampleRecord(input_file, line_number, record)
                seen[instr_type] += 1
                type_samples = samples[instr_type]
                if len(type_samples) < per_type:
                    type_samples.append(sample)
                else:
                    replacement_index = rng.randrange(seen[instr_type])
                    if replacement_index < per_type:
                        type_samples[replacement_index] = sample

    missing = [
        f"{instr_type}: found {seen[instr_type]}/{per_type}"
        for instr_type in INSTR_TYPES
        if seen[instr_type] < per_type
    ]
    if missing:
        raise ValueError(
            "Not enough records for requested instruction types: "
            + ", ".join(missing)
        )

    for instr_type in INSTR_TYPES:
        samples[instr_type].sort(
            key=lambda item: (str(item.source_file), item.source_line_number)
        )
    return samples


def _write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip("\n") + "\n", encoding="utf-8")


def export_samples(
    samples: Dict[str, List[SampleRecord]], output_dir: Path
) -> List[Path]:
    """Write sampled records into type and source-line directories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_dirs: List[Path] = []

    for instr_type in INSTR_TYPES:
        for sample in samples[instr_type]:
            record = sample.record
            record_dir = (
                output_dir
                / instr_type
                / f"line_{sample.source_line_number:06d}"
            )
            record_dir.mkdir(parents=True, exist_ok=True)
            _write_text(
                record_dir / "pre_edit.py",
                record["code_before_purify"],
            )
            _write_text(
                record_dir / "post_edit.py",
                record["code_after_purify"],
            )
            instruction = {
                "instruct_purify": record["instruct_purify"],
                "commit": record["commit"],
                "instr_type": record["instr_type"],
                "source_file": str(sample.source_file),
                "source_line_number": sample.source_line_number,
                MANUAL_REVIEW_FIELD: None,
            }
            (record_dir / "instruction.json").write_text(
                json.dumps(instruction, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            exported_dirs.append(record_dir)

    return exported_dirs


def sample_and_export(
    input_files: Sequence[Path],
    output_dir: Path,
    count: int = DEFAULT_COUNT,
    seed: int = DEFAULT_SEED,
) -> List[Path]:
    """Sample validated records, then export them only after sampling succeeds."""
    samples = sample_records(input_files, count=count, seed=seed)
    return export_samples(samples, output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export an equal, reproducible sample of the four supported "
            "DT-filtered instruction types."
        )
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        nargs="+",
        default=list(DEFAULT_INPUT_FILES),
        help="One or more DT-filtered JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Export directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=(
            f"Total sample count, divisible by {len(INSTR_TYPES)} "
            f"(default: {DEFAULT_COUNT})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED}).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        exported_dirs = sample_and_export(
            input_files=args.input_file,
            output_dir=args.output_dir,
            count=args.count,
            seed=args.seed,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    per_type = len(exported_dirs) // len(INSTR_TYPES)
    print(
        f"Exported {len(exported_dirs)} records "
        f"({per_type} per instruction type) to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
