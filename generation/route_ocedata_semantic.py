#!/usr/bin/env python3
"""Route OCEData semantic inputs by generator and assemble final outputs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, BinaryIO, Iterator

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "OCEData"
WORK_DIR = DATA_DIR / "semantic_work"

DEFAULT_MAIN_SOURCE = DATA_DIR / "ocedata.jsonl"
DEFAULT_MAIN_QUALITY = DATA_DIR / "ocedata_quality_filtered.jsonl"
DEFAULT_FT_SOURCE = DATA_DIR / "ocedataft.jsonl"
DEFAULT_FT_QUALITY = DATA_DIR / "ocedataft_quality_filtered.jsonl"
DEFAULT_QWEN_INPUT = WORK_DIR / "ocedata_quality_filtered_qwen3.jsonl"
DEFAULT_DS_INPUT = WORK_DIR / "ocedata_quality_filtered_ds.jsonl"


def _digest(raw_line: bytes) -> str:
    return hashlib.sha256(raw_line.rstrip(b"\r\n")).hexdigest()


def _json_object(raw_line: bytes, path: Path, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(raw_line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}:{line_number}: record must be a JSON object")
    return value


def _records(path: Path) -> Iterator[tuple[int, bytes, dict[str, Any]]]:
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL line")
            yield line_number, raw_line, _json_object(raw_line, path, line_number)


def _route(record: dict[str, Any], path: Path, line_number: int) -> str:
    instr_type = record.get("instr_type")
    if not isinstance(instr_type, str):
        raise ValueError(f"{path}:{line_number}: instr_type must be a string")
    if instr_type.startswith("qwen3_"):
        return "qwen3"
    if instr_type.startswith("ds_"):
        return "ds"
    raise ValueError(f"{path}:{line_number}: unsupported instr_type {instr_type!r}")


def _temporary_binary(path: Path) -> tuple[BinaryIO, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
    return handle, Path(handle.name)


def _temporary_text(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    return handle, Path(handle.name)


def _publish(handle: Any, temporary: Path, destination: Path) -> None:
    handle.flush()
    os.fsync(handle.fileno())
    handle.close()
    os.replace(temporary, destination)


def prepare(input_path: Path, qwen_path: Path, ds_path: Path) -> dict[str, Any]:
    if input_path.resolve() in {qwen_path.resolve(), ds_path.resolve()}:
        raise ValueError("Input and routed output paths must be distinct")
    if qwen_path.resolve() == ds_path.resolve():
        raise ValueError("Routed output paths must be distinct")

    qwen_handle, qwen_tmp = _temporary_binary(qwen_path)
    ds_handle, ds_tmp = _temporary_binary(ds_path)
    counts: Counter[str] = Counter()
    input_hash = hashlib.sha256()
    try:
        for line_number, raw_line, record in _records(input_path):
            input_hash.update(raw_line)
            route = _route(record, input_path, line_number)
            counts[route] += 1
            (qwen_handle if route == "qwen3" else ds_handle).write(raw_line)
        if not counts:
            raise ValueError(f"Input file is empty: {input_path}")
        _publish(qwen_handle, qwen_tmp, qwen_path)
        _publish(ds_handle, ds_tmp, ds_path)
    except BaseException:
        qwen_handle.close()
        ds_handle.close()
        qwen_tmp.unlink(missing_ok=True)
        ds_tmp.unlink(missing_ok=True)
        raise

    return {
        "input": str(input_path),
        "input_sha256": input_hash.hexdigest(),
        "total": sum(counts.values()),
        "routes": dict(sorted(counts.items())),
        "qwen_input": str(qwen_path),
        "ds_input": str(ds_path),
    }


def _source_line_map(path: Path) -> tuple[dict[str, int], str, int]:
    mapping: dict[str, int] = {}
    digest = hashlib.sha256()
    total = 0
    for line_number, raw_line, _ in _records(path):
        digest.update(raw_line)
        key = _digest(raw_line)
        if key in mapping:
            raise ValueError(f"{path}: duplicate records at lines {mapping[key]} and {line_number}")
        mapping[key] = line_number
        total += 1
    return mapping, digest.hexdigest(), total


class RouteReader:
    def __init__(self, route: str, input_path: Path, result_path: Path) -> None:
        self.route = route
        self.input_path = input_path
        self.result_path = result_path
        self.input_handle = input_path.open("rb")
        self.result_handle = result_path.open("rb")
        self.line_number = 0

    def next(self, expected_raw: bytes) -> dict[str, Any]:
        input_raw = self.input_handle.readline()
        result_raw = self.result_handle.readline()
        self.line_number += 1
        if not input_raw or not result_raw:
            raise ValueError(f"{self.route}: semantic input/results ended early")
        if _digest(input_raw) != _digest(expected_raw):
            raise ValueError(f"{self.route}:{self.line_number}: routed input order mismatch")
        result = _json_object(result_raw, self.result_path, self.line_number)
        if result.get("line_number") != self.line_number:
            raise ValueError(f"{self.result_path}:{self.line_number}: result line number mismatch")
        return result

    def finish(self) -> None:
        if self.input_handle.readline() or self.result_handle.readline():
            raise ValueError(f"{self.route}: semantic input/results contain extra records")
        self.input_handle.close()
        self.result_handle.close()


def _write_yaml_atomic(path: Path, value: dict[str, Any]) -> None:
    handle, temporary = _temporary_text(path)
    try:
        yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=True)
        _publish(handle, temporary, path)
    except BaseException:
        handle.close()
        temporary.unlink(missing_ok=True)
        raise


def _output_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(
    *,
    source_path: Path,
    source_sha256: str,
    source_total: int,
    quality_path: Path,
    quality_total: int,
    decision_counts: Counter[str],
    by_route: dict[str, Counter[str]],
    by_instr_type: dict[str, Counter[str]],
    result_path: Path,
    filtered_path: Path,
    reused: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_file": str(source_path),
        "source_sha256": source_sha256,
        "quality_filtered_file": str(quality_path),
        "counts": {
            "source_total": source_total,
            "quality_passed": quality_total,
            "quality_failed": source_total - quality_total,
            "semantic_decisions": dict(sorted(decision_counts.items())),
            "final_passed": decision_counts.get("ACCEPT", 0),
        },
        "semantic_decisions_by_route": {
            key: dict(sorted(value.items())) for key, value in sorted(by_route.items())
        },
        "semantic_decisions_by_instr_type": {
            key: dict(sorted(value.items()))
            for key, value in sorted(by_instr_type.items())
        },
        "semantic_inference_reused_from_ocedata": reused,
        "result_file": str(result_path),
        "result_sha256": _output_hash(result_path),
        "filtered_file": str(filtered_path),
        "filtered_sha256": _output_hash(filtered_path),
    }


def finalize(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    main_source_lines, main_source_sha, main_source_total = _source_line_map(args.main_source)
    ft_source_lines, ft_source_sha, ft_source_total = _source_line_map(args.ft_source)
    readers = {
        "qwen3": RouteReader("qwen3", args.qwen_input, args.qwen_results),
        "ds": RouteReader("ds", args.ds_input, args.ds_results),
    }

    main_result_handle, main_result_tmp = _temporary_text(args.main_results)
    main_filtered_handle, main_filtered_tmp = _temporary_binary(args.main_filtered)
    main_decisions: dict[str, dict[str, Any]] = {}
    main_counts: Counter[str] = Counter()
    main_by_route: dict[str, Counter[str]] = defaultdict(Counter)
    main_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    main_quality_total = 0
    try:
        for quality_line, raw_line, record in _records(args.main_quality):
            main_quality_total += 1
            key = _digest(raw_line)
            source_line = main_source_lines.get(key)
            if source_line is None:
                raise ValueError(f"{args.main_quality}:{quality_line}: not found in source")
            route = _route(record, args.main_quality, quality_line)
            result = readers[route].next(raw_line)
            decision = result.get("decision")
            if decision not in {"ACCEPT", "REJECT", "UNCERTAIN", "ERROR"}:
                raise ValueError(f"Invalid semantic decision: {decision!r}")
            combined = dict(result)
            combined["route_line_number"] = combined.pop("line_number")
            combined["line_number"] = quality_line
            combined["source_line_number"] = source_line
            combined["record_sha256"] = key
            main_result_handle.write(json.dumps(combined, ensure_ascii=False) + "\n")
            if decision == "ACCEPT":
                main_filtered_handle.write(raw_line)
            main_counts[decision] += 1
            main_by_route[route][decision] += 1
            main_by_type[str(record["instr_type"])][decision] += 1
            main_decisions[key] = combined
        for reader in readers.values():
            reader.finish()
        _publish(main_result_handle, main_result_tmp, args.main_results)
        _publish(main_filtered_handle, main_filtered_tmp, args.main_filtered)
    except BaseException:
        main_result_handle.close()
        main_filtered_handle.close()
        main_result_tmp.unlink(missing_ok=True)
        main_filtered_tmp.unlink(missing_ok=True)
        raise

    ft_result_handle, ft_result_tmp = _temporary_text(args.ft_results)
    ft_filtered_handle, ft_filtered_tmp = _temporary_binary(args.ft_filtered)
    ft_counts: Counter[str] = Counter()
    ft_by_route: dict[str, Counter[str]] = defaultdict(Counter)
    ft_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    ft_quality_total = 0
    try:
        for quality_line, raw_line, record in _records(args.ft_quality):
            ft_quality_total += 1
            key = _digest(raw_line)
            source_line = ft_source_lines.get(key)
            if source_line is None:
                raise ValueError(f"{args.ft_quality}:{quality_line}: not found in FT source")
            main_result = main_decisions.get(key)
            if main_result is None:
                raise ValueError(f"{args.ft_quality}:{quality_line}: no OCEData semantic decision")
            combined = dict(main_result)
            combined["ocedata_quality_filtered_line_number"] = combined["line_number"]
            combined["ocedata_source_line_number"] = combined["source_line_number"]
            combined["line_number"] = quality_line
            combined["source_line_number"] = source_line
            combined["semantic_inference_reused_from_ocedata"] = True
            ft_result_handle.write(json.dumps(combined, ensure_ascii=False) + "\n")
            decision = str(combined["decision"])
            if decision == "ACCEPT":
                ft_filtered_handle.write(raw_line)
            route = _route(record, args.ft_quality, quality_line)
            ft_counts[decision] += 1
            ft_by_route[route][decision] += 1
            ft_by_type[str(record["instr_type"])][decision] += 1
        _publish(ft_result_handle, ft_result_tmp, args.ft_results)
        _publish(ft_filtered_handle, ft_filtered_tmp, args.ft_filtered)
    except BaseException:
        ft_result_handle.close()
        ft_filtered_handle.close()
        ft_result_tmp.unlink(missing_ok=True)
        ft_filtered_tmp.unlink(missing_ok=True)
        raise


    main_summary = _summary(
        source_path=args.main_source,
        source_sha256=main_source_sha,
        source_total=main_source_total,
        quality_path=args.main_quality,
        quality_total=main_quality_total,
        decision_counts=main_counts,
        by_route=main_by_route,
        by_instr_type=main_by_type,
        result_path=args.main_results,
        filtered_path=args.main_filtered,
        reused=False,
    )
    main_summary["semantic_inputs"] = {
        "qwen3": str(args.qwen_input),
        "ds": str(args.ds_input),
    }
    main_summary["semantic_result_sources"] = {
        "qwen3": str(args.qwen_results),
        "ds": str(args.ds_results),
    }
    ft_summary = _summary(
        source_path=args.ft_source,
        source_sha256=ft_source_sha,
        source_total=ft_source_total,
        quality_path=args.ft_quality,
        quality_total=ft_quality_total,
        decision_counts=ft_counts,
        by_route=ft_by_route,
        by_instr_type=ft_by_type,
        result_path=args.ft_results,
        filtered_path=args.ft_filtered,
        reused=True,
    )
    _write_yaml_atomic(args.main_summary, main_summary)
    _write_yaml_atomic(args.ft_summary, ft_summary)
    return main_summary, ft_summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--input", type=Path, default=DEFAULT_MAIN_QUALITY)
    prepare_parser.add_argument("--qwen-input", type=Path, default=DEFAULT_QWEN_INPUT)
    prepare_parser.add_argument("--ds-input", type=Path, default=DEFAULT_DS_INPUT)

    finalize_parser = commands.add_parser("finalize")
    finalize_parser.add_argument("--main-source", type=Path, default=DEFAULT_MAIN_SOURCE)
    finalize_parser.add_argument("--main-quality", type=Path, default=DEFAULT_MAIN_QUALITY)
    finalize_parser.add_argument("--ft-source", type=Path, default=DEFAULT_FT_SOURCE)
    finalize_parser.add_argument("--ft-quality", type=Path, default=DEFAULT_FT_QUALITY)
    finalize_parser.add_argument("--qwen-input", type=Path, default=DEFAULT_QWEN_INPUT)
    finalize_parser.add_argument("--ds-input", type=Path, default=DEFAULT_DS_INPUT)
    finalize_parser.add_argument(
        "--qwen-results",
        type=Path,
        default=WORK_DIR / "ocedata_quality_filtered_qwen3_semantic_results.jsonl",
    )
    finalize_parser.add_argument(
        "--ds-results",
        type=Path,
        default=WORK_DIR / "ocedata_quality_filtered_ds_semantic_results.jsonl",
    )
    finalize_parser.add_argument(
        "--main-results",
        type=Path,
        default=DATA_DIR / "ocedata_quality_semantic_results.jsonl",
    )
    finalize_parser.add_argument(
        "--main-filtered",
        type=Path,
        default=DATA_DIR / "ocedata_quality_semantic_filtered.jsonl",
    )
    finalize_parser.add_argument(
        "--main-summary",
        type=Path,
        default=DATA_DIR / "ocedata_quality_semantic_summary.yaml",
    )
    finalize_parser.add_argument(
        "--ft-results",
        type=Path,
        default=DATA_DIR / "ocedataft_quality_semantic_results.jsonl",
    )
    finalize_parser.add_argument(
        "--ft-filtered",
        type=Path,
        default=DATA_DIR / "ocedataft_quality_semantic_filtered.jsonl",
    )
    finalize_parser.add_argument(
        "--ft-summary",
        type=Path,
        default=DATA_DIR / "ocedataft_quality_semantic_summary.yaml",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        print(yaml.safe_dump(prepare(args.input, args.qwen_input, args.ds_input), sort_keys=False))
    else:
        main_summary, ft_summary = finalize(args)
        print(yaml.safe_dump({"ocedata": main_summary["counts"], "ocedataft": ft_summary["counts"]}, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
