#!/usr/bin/env python3
"""Split paired descriptive/lazy instructions into independent JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterator, Sequence


DEFAULT_DESCRIPTIVE_FIELD = "instruct_descriptive_purify"
DEFAULT_LAZY_FIELD = "instruct_lazy_purify"
MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _default_output_path(input_file: Path, variant: str) -> Path:
    return input_file.with_name(f"{input_file.stem}_{variant}.jsonl")


def _validate_model_name(model_name: str) -> str:
    if not isinstance(model_name, str) or not MODEL_NAME_PATTERN.fullmatch(model_name):
        raise ValueError(
            "model_name must be a non-empty label containing only ASCII letters, "
            "digits, '.', '_', or '-', and must start with a letter or digit"
        )
    return model_name


def _validate_field_names(descriptive_field: str, lazy_field: str) -> None:
    for name, value in (
        ("descriptive_field", descriptive_field),
        ("lazy_field", lazy_field),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    if descriptive_field == lazy_field:
        raise ValueError("descriptive_field and lazy_field must be different")


def _validate_paths(
    input_file: Path, descriptive_file: Path, lazy_file: Path
) -> None:
    resolved = {
        input_file.resolve(),
        descriptive_file.resolve(),
        lazy_file.resolve(),
    }
    if len(resolved) != 3:
        raise ValueError("Input, descriptive output, and lazy output must be distinct")
    for path in (descriptive_file, lazy_file):
        if path.exists():
            raise ValueError(f"Output path already exists: {path}")


def _records(input_file: Path) -> Iterator[tuple[int, str, dict[str, Any]]]:
    try:
        input_handle = input_file.open("r", encoding="utf-8", newline="")
    except OSError as error:
        raise OSError(f"Cannot open input file {input_file}: {error}") from error

    with input_handle:
        for line_number, raw_line in enumerate(input_handle, start=1):
            if not raw_line.strip():
                raise ValueError(f"{input_file}:{line_number}: blank JSONL line")
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{input_file}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"{input_file}:{line_number}: record must be a JSON object"
                )
            yield line_number, raw_line, record


def _required_instruction(
    record: dict[str, Any], field: str, input_file: Path, line_number: int
) -> str:
    if field not in record:
        raise ValueError(
            f"{input_file}:{line_number}: missing instruction field {field!r}"
        )
    value = record[field]
    if not isinstance(value, str):
        raise ValueError(
            f"{input_file}:{line_number}: instruction field {field!r} "
            "must be a string"
        )
    if not value.strip():
        raise ValueError(
            f"{input_file}:{line_number}: instruction field {field!r} "
            "must not be empty"
        )
    return value


def _validate_input(
    input_file: Path, descriptive_field: str, lazy_field: str
) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    for line_number, raw_line, record in _records(input_file):
        digest.update(raw_line.encode("utf-8"))
        _required_instruction(record, descriptive_field, input_file, line_number)
        _required_instruction(record, lazy_field, input_file, line_number)
        total += 1
    if total == 0:
        raise ValueError(f"Input file is empty: {input_file}")
    return total, digest.hexdigest()


def _write_split_records(
    *,
    input_file: Path,
    descriptive_handle: Any,
    lazy_handle: Any,
    descriptive_field: str,
    lazy_field: str,
    model_name: str,
) -> str:
    digest = hashlib.sha256()
    for line_number, raw_line, record in _records(input_file):
        digest.update(raw_line.encode("utf-8"))
        descriptive_instruction = _required_instruction(
            record, descriptive_field, input_file, line_number
        )
        lazy_instruction = _required_instruction(
            record, lazy_field, input_file, line_number
        )

        descriptive_record = dict(record)
        descriptive_record["instruct_purify"] = descriptive_instruction
        descriptive_record["instr_type"] = f"{model_name}_descriptive"
        descriptive_handle.write(
            json.dumps(descriptive_record, ensure_ascii=False) + "\n"
        )

        lazy_record = dict(record)
        lazy_record["instruct_purify"] = lazy_instruction
        lazy_record["instr_type"] = f"{model_name}_lazy"
        lazy_handle.write(json.dumps(lazy_record, ensure_ascii=False) + "\n")
    return digest.hexdigest()


def split_instructions(
    *,
    input_file: str | Path,
    model_name: str,
    descriptive_file: str | Path | None = None,
    lazy_file: str | Path | None = None,
    descriptive_field: str = DEFAULT_DESCRIPTIVE_FIELD,
    lazy_field: str = DEFAULT_LAZY_FIELD,
) -> dict[str, Any]:
    """Validate and split every source record into two normalized views."""

    input_file = Path(input_file).resolve()
    descriptive_file = Path(
        descriptive_file or _default_output_path(input_file, "descriptive")
    ).resolve()
    lazy_file = Path(
        lazy_file or _default_output_path(input_file, "lazy")
    ).resolve()
    model_name = _validate_model_name(model_name)
    _validate_field_names(descriptive_field, lazy_field)
    _validate_paths(input_file, descriptive_file, lazy_file)
    total, validated_sha256 = _validate_input(
        input_file, descriptive_field, lazy_field
    )

    descriptive_file.parent.mkdir(parents=True, exist_ok=True)
    lazy_file.parent.mkdir(parents=True, exist_ok=True)
    descriptive_temporary: Path | None = None
    lazy_temporary: Path | None = None
    committed: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=descriptive_file.parent,
            prefix=f".{descriptive_file.name}.",
            suffix=".tmp",
            delete=False,
        ) as descriptive_handle:
            descriptive_temporary = Path(descriptive_handle.name)
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=lazy_file.parent,
                prefix=f".{lazy_file.name}.",
                suffix=".tmp",
                delete=False,
            ) as lazy_handle:
                lazy_temporary = Path(lazy_handle.name)
                written_sha256 = _write_split_records(
                    input_file=input_file,
                    descriptive_handle=descriptive_handle,
                    lazy_handle=lazy_handle,
                    descriptive_field=descriptive_field,
                    lazy_field=lazy_field,
                    model_name=model_name,
                )
                descriptive_handle.flush()
                lazy_handle.flush()
                os.fsync(descriptive_handle.fileno())
                os.fsync(lazy_handle.fileno())

        if written_sha256 != validated_sha256:
            raise ValueError("Input file changed while instructions were being split")
        for path in (descriptive_file, lazy_file):
            if path.exists():
                raise ValueError(f"Output path appeared during splitting: {path}")
        os.replace(descriptive_temporary, descriptive_file)
        committed.append(descriptive_file)
        descriptive_temporary = None
        os.replace(lazy_temporary, lazy_file)
        committed.append(lazy_file)
        lazy_temporary = None
    except Exception:
        for path in committed:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for temporary_path in (descriptive_temporary, lazy_temporary):
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass

    return {
        "total": total,
        "model_name": model_name,
        "descriptive_instr_type": f"{model_name}_descriptive",
        "lazy_instr_type": f"{model_name}_lazy",
        "input_file": str(input_file),
        "input_sha256": validated_sha256,
        "descriptive_file": str(descriptive_file),
        "lazy_file": str(lazy_file),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split paired descriptive and lazy edit instructions into two "
            "independent semantic-check inputs."
        )
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--descriptive-file", type=Path)
    parser.add_argument("--lazy-file", type=Path)
    parser.add_argument(
        "--descriptive-field", default=DEFAULT_DESCRIPTIVE_FIELD
    )
    parser.add_argument("--lazy-field", default=DEFAULT_LAZY_FIELD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = split_instructions(
            input_file=args.input_file,
            model_name=args.model_name,
            descriptive_file=args.descriptive_file,
            lazy_file=args.lazy_file,
            descriptive_field=args.descriptive_field,
            lazy_field=args.lazy_field,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"Records split: {summary['total']}")
    print(f"Descriptive data: {summary['descriptive_file']}")
    print(f"Lazy data: {summary['lazy_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
