#!/usr/bin/env python3
"""Export the first N JSONL edit samples into human-readable files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List, Optional, Sequence


DEFAULT_INPUT_FILE = Path("data/OCEData/ocedataft_quality_filtered.jsonl")


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


def _write_jsonl(path: Path, record: dict) -> None:
    """Write one JSON object as a UTF-8 JSONL record."""
    path.write_text(
        json.dumps(record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def export_first_samples(
    input_file: Path,
    count: int,
    output_dir: Path,
    pre_field: str = "code_before_purify",
    post_field: str = "code_after_purify",
    instruction_field: str = "instruct_purify",
) -> List[Path]:
    """Export the first ``count`` records into one directory per JSONL line."""
    if count <= 0:
        raise ValueError("count must be a positive integer")
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    exported_dirs: List[Path] = []

    with input_file.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if line_number > count:
                break
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {input_file} at line {line_number}: {exc}"
                ) from exc

            pre_code = _required_string(record, pre_field, line_number)
            post_code = _required_string(record, post_field, line_number)
            instruction = _required_string(record, instruction_field, line_number)
            commit = _required_string(record, "commit", line_number)
            instr_type = _required_string(record, "instr_type", line_number)

            record_dir = output_dir / f"line_{line_number:06d}"
            record_dir.mkdir(parents=True, exist_ok=True)
            _write_text(record_dir / "pre_edit.py", pre_code)
            _write_text(record_dir / "post_edit.py", post_code)
            _write_jsonl(
                record_dir / "instruction.jsonl",
                {
                    instruction_field: instruction,
                    "commit": commit,
                    "instr_type": instr_type,
                },
            )
            # Remove the file produced by older versions when reusing an output dir.
            (record_dir / "instruction.txt").unlink(missing_ok=True)
            exported_dirs.append(record_dir)

    if len(exported_dirs) < count:
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
        help="Number of records to export from the beginning of the input file.",
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
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Exported {len(exported_dirs)} records to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
