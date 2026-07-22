#!/usr/bin/env python3
"""Export the first N JSONL edit samples into human-readable files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List, Optional, Sequence


DEFAULT_INPUT_FILE = Path("data/OCEData/ocedataft_quality_filtered.jsonl")

MANUAL_REVIEW_FIELDS = (
    # pre-edit code 是否构成合理程序。
    "manual_pre_edit_is_reasonable_program",
    # pre-edit 是否确实尚未满足 instruction。
    "manual_pre_edit_does_not_satisfy_instruction",
    # instruction 是否明确且可实施。
    "manual_instruction_is_clear_and_actionable",
    # post-edit code 是否构成合理程序。
    "manual_post_edit_is_reasonable_program",
    # post-edit 是否完成了编辑指令。
    "manual_post_edit_fulfills_instruction",
    # post-edit 是否引入了与编辑指令无关的修改。
    "manual_post_edit_has_unrelated_changes",
    # post-edit 是否引入了 pre-edit 中不存在的新缺陷。
    "manual_post_edit_has_new_defects",
)


def default_output_dir(input_file: Path) -> Path:
    """Return the default output directory beside the input JSONL file."""
    return input_file.with_name(f"{input_file.stem}_samples")


def _required_string(record: object, field: str, line_number: int) -> str:
    if not isinstance(record, dict):
        raise ValueError(f"Expected a JSON object at line {line_number}")
    if field not in record:
        raise ValueError(f"Line {line_number} is missing required field '{field}'")
    value = record[field]
    if not isinstance(value, str):
        raise ValueError(f"Line {line_number} field '{field}' must be a string")
    return value


def _write_text(path: Path, content: str) -> None:
    """Write text with one trailing newline for convenient terminal viewing."""
    path.write_text(content.rstrip("\n") + "\n", encoding="utf-8")


def _write_json(path: Path, record: dict) -> None:
    """Write one human-readable UTF-8 JSON object."""
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _validate_instr_types(instr_types: Sequence[str]) -> List[str]:
    """Validate type names before using them as output directory names."""
    unique_types = list(dict.fromkeys(instr_types))
    if len(unique_types) != len(instr_types):
        raise ValueError("instr_type values must not be repeated")
    for instr_type in unique_types:
        if (
            not instr_type
            or instr_type in {".", ".."}
            or Path(instr_type).name != instr_type
        ):
            raise ValueError(
                f"instr_type '{instr_type}' cannot be used as a directory name"
            )
    return unique_types


def _export_record(
    record: object,
    line_number: int,
    record_dir: Path,
    pre_field: str,
    post_field: str,
    instruction_field: str,
) -> None:
    """Write one edit record to its human-readable output directory."""
    pre_code = _required_string(record, pre_field, line_number)
    post_code = _required_string(record, post_field, line_number)
    instruction = _required_string(record, instruction_field, line_number)
    commit = _required_string(record, "commit", line_number)
    instr_type = _required_string(record, "instr_type", line_number)

    record_dir.mkdir(parents=True, exist_ok=True)
    _write_text(record_dir / "pre_edit.py", pre_code)
    _write_text(record_dir / "post_edit.py", post_code)
    instruction_record = {
        instruction_field: instruction,
        "commit": commit,
        "instr_type": instr_type,
    }
    instruction_record.update(
        {field: None for field in MANUAL_REVIEW_FIELDS}
    )
    _write_json(
        record_dir / "instruction.json",
        instruction_record,
    )
    # Remove files produced by older versions when reusing an output directory.
    (record_dir / "instruction.jsonl").unlink(missing_ok=True)
    (record_dir / "instruction.txt").unlink(missing_ok=True)


def export_first_samples(
    input_file: Path,
    count: int,
    output_dir: Path,
    pre_field: str = "code_before_purify",
    post_field: str = "code_after_purify",
    instruction_field: str = "instruct_purify",
    instr_types: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Export the first records overall or the first ``count`` of each type."""
    if count <= 0:
        raise ValueError("count must be a positive integer")
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    selected_types = _validate_instr_types(instr_types or [])
    remaining = {instr_type: count for instr_type in selected_types}
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_dirs: List[Path] = []

    with input_file.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not selected_types and line_number > count:
                break
            if selected_types and not any(remaining.values()):
                break
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {input_file} at line {line_number}: {exc}"
                ) from exc

            if selected_types:
                record_type = _required_string(record, "instr_type", line_number)
                if record_type not in remaining or remaining[record_type] == 0:
                    continue
                record_dir = (
                    output_dir / record_type / f"line_{line_number:06d}"
                )
                remaining[record_type] -= 1
            else:
                record_dir = output_dir / f"line_{line_number:06d}"

            _export_record(
                record,
                line_number,
                record_dir,
                pre_field,
                post_field,
                instruction_field,
            )
            exported_dirs.append(record_dir)

    missing_types = {
        instr_type: missing
        for instr_type, missing in remaining.items()
        if missing
    }
    if missing_types:
        details = ", ".join(
            f"{instr_type}: found {count - missing}/{count}"
            for instr_type, missing in missing_types.items()
        )
        raise ValueError(f"Not enough records for requested instr_type values: {details}")
    if not selected_types and len(exported_dirs) < count:
        raise ValueError(
            f"Requested {count} records, but {input_file} contains only "
            f"{len(exported_dirs)} lines"
        )
    return exported_dirs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export the first N pre-edit snippets, post-edit snippets, and edit "
            "instructions from a JSONL dataset."
        )
    )
    parser.add_argument(
        "n",
        type=int,
        help=(
            "Number of records to export, or number per type when --instr-type "
            "is used."
        ),
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=f"Input JSONL file (default: {DEFAULT_INPUT_FILE}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <input_stem>_samples beside input).",
    )
    parser.add_argument("--pre-field", default="code_before_purify")
    parser.add_argument("--post-field", default="code_after_purify")
    parser.add_argument("--instruction-field", default="instruct_purify")
    parser.add_argument(
        "--instr-type",
        nargs="+",
        default=None,
        help=(
            "Export the first N records of each listed instr_type into "
            "type-named subdirectories."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or default_output_dir(args.input_file)
    try:
        exported_dirs = export_first_samples(
            input_file=args.input_file,
            count=args.n,
            output_dir=output_dir,
            pre_field=args.pre_field,
            post_field=args.post_field,
            instruction_field=args.instruction_field,
            instr_types=args.instr_type,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Exported {len(exported_dirs)} records to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
