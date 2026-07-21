#!/usr/bin/env python3
"""Export pre-edit and post-edit code for selected quality issue records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Set


ISSUES_SUFFIX = "_quality_issues"


def infer_input_file(issues_file: Path) -> Path:
    """Infer the source JSONL path from a standard quality report name."""
    if not issues_file.stem.endswith(ISSUES_SUFFIX):
        raise ValueError(
            "Cannot infer input file: the issues file name must end with "
            f"'{ISSUES_SUFFIX}{issues_file.suffix}', or --input-file must be set"
        )
    input_stem = issues_file.stem[: -len(ISSUES_SUFFIX)]
    return issues_file.with_name(f"{input_stem}{issues_file.suffix}")


def default_output_dir(input_file: Path) -> Path:
    """Return the default sibling directory for exported issue code."""
    return input_file.with_name(f"{input_file.stem}_quality_issue_code")


def load_selected_issues(
    issues_file: Path, requested_lines: Set[int]
) -> Dict[int, Dict[str, object]]:
    """Load report entries for the requested one-based source line numbers."""
    selected: Dict[int, Dict[str, object]] = {}
    with issues_file.open("r", encoding="utf-8") as handle:
        for report_line, raw_line in enumerate(handle, 1):
            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {issues_file} at report line {report_line}: {exc}"
                ) from exc
            line_number = item.get("line_number")
            if line_number in requested_lines:
                if line_number in selected:
                    raise ValueError(
                        f"Duplicate line_number {line_number} in {issues_file}"
                    )
                selected[line_number] = item

    missing = sorted(requested_lines - selected.keys())
    if missing:
        raise ValueError(
            "Requested line_number values are absent from the quality report: "
            + ", ".join(map(str, missing))
        )
    return selected


def load_source_records(
    input_file: Path, requested_lines: Set[int]
) -> Dict[int, Dict[str, object]]:
    """Stream the source JSONL and retain only requested records."""
    selected: Dict[int, Dict[str, object]] = {}
    with input_file.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if line_number not in requested_lines:
                continue
            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {input_file} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(item, dict):
                raise ValueError(
                    f"Expected a JSON object in {input_file} at line {line_number}"
                )
            selected[line_number] = item
            if len(selected) == len(requested_lines):
                break

    missing = sorted(requested_lines - selected.keys())
    if missing:
        raise ValueError(
            "Requested line_number values exceed or are absent from the input file: "
            + ", ".join(map(str, missing))
        )
    return selected


def _string_field(record: Dict[str, object], field_name: str, line_number: int) -> str:
    if field_name not in record:
        raise ValueError(
            f"Source line {line_number} is missing required field '{field_name}'"
        )
    value = record[field_name]
    if not isinstance(value, str):
        raise ValueError(
            f"Source line {line_number} field '{field_name}' must be a string"
        )
    return value


def export_issue_code(
    input_file: Path,
    issues_file: Path,
    line_numbers: Sequence[int],
    output_dir: Path,
    pre_field: str = "code_before_purify",
    post_field: str = "code_after_purify",
    instruction_field: str = "instruct_purify",
) -> List[Path]:
    """Export selected issue records into per-line code directories."""
    requested_lines = set(line_numbers)
    if not requested_lines or any(number <= 0 for number in requested_lines):
        raise ValueError("line_number values must be positive integers")
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not issues_file.is_file():
        raise FileNotFoundError(f"Quality issues file not found: {issues_file}")

    issue_records = load_selected_issues(issues_file, requested_lines)
    source_records = load_source_records(input_file, requested_lines)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_dirs = []
    width = max(6, len(str(max(requested_lines))))
    for line_number in sorted(requested_lines):
        record = source_records[line_number]
        record_dir = output_dir / f"line_{line_number:0{width}d}"
        record_dir.mkdir(parents=True, exist_ok=True)

        pre_code = _string_field(record, pre_field, line_number)
        post_code = _string_field(record, post_field, line_number)
        edit_instruction = _string_field(record, instruction_field, line_number)
        metadata = dict(issue_records[line_number])
        metadata["edit_instruction"] = edit_instruction
        (record_dir / "pre_edit.py").write_text(pre_code, encoding="utf-8")
        (record_dir / "post_edit.py").write_text(post_code, encoding="utf-8")
        (record_dir / "issues.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        exported_dirs.append(record_dir)

    return exported_dirs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export pre-edit and post-edit code for quality issue lines."
    )
    parser.add_argument("--issues-file", type=Path, required=True)
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Source JSONL path; inferred from --issues-file when omitted.",
    )
    parser.add_argument(
        "--line-number",
        type=int,
        nargs="+",
        required=True,
        help="One or more one-based source line numbers from the issue report.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Export directory (default: <input_stem>_quality_issue_code).",
    )
    parser.add_argument("--pre-field", default="code_before_purify")
    parser.add_argument("--post-field", default="code_after_purify")
    parser.add_argument("--instruction-field", default="instruct_purify")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        input_file = args.input_file or infer_input_file(args.issues_file)
        output_dir = args.output_dir or default_output_dir(input_file)
        exported_dirs = export_issue_code(
            input_file=input_file,
            issues_file=args.issues_file,
            line_numbers=args.line_number,
            output_dir=output_dir,
            pre_field=args.pre_field,
            post_field=args.post_field,
            instruction_field=args.instruction_field,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for directory in exported_dirs:
        print(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
