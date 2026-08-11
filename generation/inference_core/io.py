from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import yaml


class InferenceError(RuntimeError):
    """Raised when an inference run cannot be safely advanced."""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    records: list[dict[str, Any]] = []
    try:
        input_file = path.open("r", encoding="utf-8")
    except OSError as error:
        raise InferenceError(f"Cannot open JSONL file {path}: {error}") from error

    with input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                raise InferenceError(f"{path}:{line_number}: blank JSONL line")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise InferenceError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise InferenceError(f"{path}:{line_number}: record must be an object")
            records.append(record)
    return records


def repair_incomplete_jsonl_tail(path: str | Path) -> Path | None:
    """Remove and preserve a single incomplete final JSONL fragment.

    A malformed record is only repairable when it is the unterminated final
    line. Blank lines, malformed complete lines, and corruption before the
    final line remain hard errors handled by ``read_jsonl``.
    """

    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    data = path.read_bytes()
    if data.endswith(b"\n"):
        return None
    line_start = data.rfind(b"\n") + 1
    fragment = data[line_start:]
    if not fragment.strip():
        return None
    try:
        text = fragment.decode("utf-8")
        json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    else:
        return None

    suffix = 1
    while True:
        preserved_path = path.with_name(
            f"{path.name}.corrupt_tail_{suffix:03d}"
        )
        try:
            with preserved_path.open("xb") as preserved_file:
                preserved_file.write(fragment)
                preserved_file.flush()
                os.fsync(preserved_file.fileno())
            break
        except FileExistsError:
            suffix += 1

    with path.open("r+b") as output_file:
        output_file.truncate(line_start)
        output_file.flush()
        os.fsync(output_file.fileno())
    return preserved_path


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def atomic_write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as input_file:
            value = yaml.safe_load(input_file) or {}
    except OSError as error:
        raise InferenceError(f"Cannot open YAML file {path}: {error}") from error
    if not isinstance(value, dict):
        raise InferenceError(f"YAML file must contain a mapping: {path}")
    return value


def atomic_write_yaml(path: str | Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            yaml.safe_dump(value, output_file, allow_unicode=True, sort_keys=False)
            output_file.flush()
            os.fsync(output_file.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
