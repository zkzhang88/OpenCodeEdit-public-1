#!/usr/bin/env python3
"""Run resumable semantic checks through the shared inference framework."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

if __package__:
    from .inference_core import (
        continue_run,
        create_run,
        resume_run,
        retry_run,
        show_status,
    )
    from .inference_core.io import (
        InferenceError,
        atomic_write_jsonl,
        atomic_write_yaml,
        load_yaml,
        read_jsonl,
        sha256_file,
    )
    from .prompts_for_check import get_prompts
else:
    from inference_core import (  # type: ignore[no-redef]
        continue_run,
        create_run,
        resume_run,
        retry_run,
        show_status,
    )
    from inference_core.io import (  # type: ignore[no-redef]
        InferenceError,
        atomic_write_jsonl,
        atomic_write_yaml,
        load_yaml,
        read_jsonl,
        sha256_file,
    )
    from prompts_for_check import get_prompts  # type: ignore[no-redef]


EXECUTORS = ("api", "siliconflow-batch", "llm-infer")
MANIFEST_NAME = "semantic_manifest.yaml"
STATE_NAME = "semantic_state.jsonl"
MANIFEST_SCHEMA_VERSION = 1
VERDICTS = {"PASS", "FAIL", "UNCERTAIN"}
CHECK_KEYS = (
    "pre_not_satisfied",
    "post_fulfills",
    "no_unrelated_changes",
)
PLACEHOLDERS = (
    "{instruction}",
    "{pre_edit_code}",
    "{post_edit_code}",
    "{diff}",
)


class ResponseValidationError(ValueError):
    """Raised when a model response does not match the semantic protocol."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _semantic_report(message: str) -> None:
    print(f"[semantic] {message}", file=sys.stderr, flush=True)


def validate_prompt_template(user_prompt: str) -> None:
    """Require every supported placeholder exactly once."""

    invalid = [item for item in PLACEHOLDERS if user_prompt.count(item) != 1]
    if invalid:
        raise InferenceError(
            "User prompt must contain each placeholder exactly once; invalid: "
            + ", ".join(invalid)
        )


def make_diff(pre_edit_code: str, post_edit_code: str) -> str:
    """Return a deterministic unified diff for the two code versions."""

    return "\n".join(
        difflib.unified_diff(
            pre_edit_code.splitlines(),
            post_edit_code.splitlines(),
            fromfile="pre_edit.py",
            tofile="post_edit.py",
            lineterm="",
        )
    )


def render_user_prompt(
    template: str,
    instruction: str,
    pre_edit_code: str,
    post_edit_code: str,
    diff: str,
) -> str:
    """Render without interpreting braces in code or the JSON response example."""

    validate_prompt_template(template)
    replacements = {
        "{instruction}": instruction,
        "{pre_edit_code}": pre_edit_code,
        "{post_edit_code}": post_edit_code,
        "{diff}": diff,
    }
    rendered = template
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def _require_exact_keys(value: object, expected: set[str], location: str) -> dict:
    if not isinstance(value, dict):
        raise ResponseValidationError(f"{location} must be a JSON object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ResponseValidationError(
            f"{location} has invalid keys; missing={missing}, extra={extra}"
        )
    return value


def validate_response(raw_response: str, finish_reason: str | None = None) -> dict:
    """Parse and strictly validate one semantic-check response."""

    if finish_reason == "length":
        raise ResponseValidationError("Model response was truncated")
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise ResponseValidationError("Model returned an empty response")
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as error:
        raise ResponseValidationError(
            f"Model response is not valid JSON: {error}"
        ) from error

    parsed = _require_exact_keys(parsed, set(CHECK_KEYS), "response")
    for key in CHECK_KEYS:
        expected = {"verdict", "reason"}
        if key == "no_unrelated_changes":
            expected.add("unrelated_changes")
        check = _require_exact_keys(parsed[key], expected, key)
        if check["verdict"] not in VERDICTS:
            raise ResponseValidationError(
                f"{key}.verdict must be PASS, FAIL, or UNCERTAIN"
            )
        if not isinstance(check["reason"], str) or not check["reason"].strip():
            raise ResponseValidationError(f"{key}.reason must be a non-empty string")
        if key == "no_unrelated_changes" and not isinstance(
            check["unrelated_changes"], list
        ):
            raise ResponseValidationError(
                "no_unrelated_changes.unrelated_changes must be a list"
            )
    return parsed


def decide(check: dict) -> str:
    """Derive the filtering decision from the three verdicts."""

    verdicts = [check[key]["verdict"] for key in CHECK_KEYS]
    if all(verdict == "PASS" for verdict in verdicts):
        return "ACCEPT"
    if any(verdict == "FAIL" for verdict in verdicts):
        return "REJECT"
    return "UNCERTAIN"


def _prompt_hash(system_prompt: str, user_prompt: str) -> str:
    digest = hashlib.sha256()
    digest.update(system_prompt.encode("utf-8"))
    digest.update(b"\0")
    digest.update(user_prompt.encode("utf-8"))
    return digest.hexdigest()


def _required_input(record: dict[str, Any], field: str, line_number: int) -> str:
    if field not in record:
        raise InferenceError(
            f"Input line {line_number} is missing required field {field!r}"
        )
    value = record[field]
    if not isinstance(value, str):
        raise InferenceError(
            f"Input line {line_number} field {field!r} must be a string"
        )
    if not value.strip():
        raise InferenceError(
            f"Input line {line_number} field {field!r} must not be empty"
        )
    return value


def _check_delimiter_collision(
    instruction: str,
    pre_edit_code: str,
    post_edit_code: str,
    diff: str,
    line_number: int,
) -> None:
    pairs = (
        (instruction, "</INSTRUCTION>"),
        (pre_edit_code, "</PRE_EDIT_CODE>"),
        (post_edit_code, "</POST_EDIT_CODE>"),
        (diff, "</DIFF>"),
    )
    collisions = [delimiter for value, delimiter in pairs if delimiter in value]
    if collisions:
        raise InferenceError(
            f"Input line {line_number} contains prompt closing delimiter(s): "
            + ", ".join(collisions)
        )


def _load_and_prepare_input(
    input_path: Path,
    *,
    system_prompt: str,
    user_prompt: str,
    pre_field: str,
    post_field: str,
    instruction_field: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validate_prompt_template(user_prompt)
    records = read_jsonl(input_path)
    if not records:
        raise InferenceError(f"Input file is empty: {input_path}")

    prompts: list[dict[str, Any]] = []
    for line_number, record in enumerate(records, start=1):
        pre_edit_code = _required_input(record, pre_field, line_number)
        post_edit_code = _required_input(record, post_field, line_number)
        instruction = _required_input(record, instruction_field, line_number)
        diff = make_diff(pre_edit_code, post_edit_code)
        _check_delimiter_collision(
            instruction, pre_edit_code, post_edit_code, diff, line_number
        )
        prompts.append(
            {
                "prompt_id": line_number,
                "system": system_prompt,
                "user": render_user_prompt(
                    user_prompt,
                    instruction,
                    pre_edit_code,
                    post_edit_code,
                    diff,
                ),
                "source_line_number": line_number,
                "commit": record.get("commit"),
                "instr_type": record.get("instr_type"),
            }
        )
    return records, prompts


def _default_output_paths(input_path: Path) -> tuple[Path, Path, Path]:
    stem = input_path.stem
    parent = input_path.parent
    return (
        parent / f"{stem}_semantic_results.jsonl",
        parent / f"{stem}_semantic_filtered.jsonl",
        parent / f"{stem}_semantic_summary.yaml",
    )


def _validate_new_paths(
    input_path: Path,
    run_dir: Path,
    result_path: Path,
    filtered_path: Path,
    summary_path: Path,
) -> None:
    paths = [input_path, result_path, filtered_path, summary_path]
    if len({path.resolve() for path in paths}) != len(paths):
        raise InferenceError("Input and output paths must all be distinct")
    for path in (result_path, filtered_path, summary_path):
        if path.exists():
            raise InferenceError(f"Output path already exists: {path}")
    if run_dir.exists():
        if not run_dir.is_dir():
            raise InferenceError(f"Run directory is not a directory: {run_dir}")
        if any(run_dir.iterdir()):
            raise InferenceError(f"Run directory is not empty: {run_dir}")


def _manifest_path(run_dir: str | Path) -> Path:
    return Path(run_dir).resolve() / MANIFEST_NAME


def _save_manifest(manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _now()
    atomic_write_yaml(_manifest_path(manifest["run_dir"]), manifest)


def _load_manifest(run_dir: str | Path) -> dict[str, Any]:
    manifest = load_yaml(_manifest_path(run_dir))
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise InferenceError(
            "Unsupported semantic manifest schema version; create a new run"
        )
    return manifest


def _validate_snapshots(manifest: dict[str, Any]) -> None:
    input_path = Path(manifest["input_path"])
    if sha256_file(input_path) != manifest["input_sha256"]:
        raise InferenceError("Input file changed since the semantic run was created")
    if sha256_file(manifest["config_path"]) != manifest["config_sha256"]:
        raise InferenceError(
            "Inference config changed since the semantic run was created"
        )
    system_prompt, user_prompt = get_prompts()
    if _prompt_hash(system_prompt, user_prompt) != manifest["prompt_sha256"]:
        raise InferenceError("Semantic prompts changed since the run was created")


def _initial_state(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "line_number": line_number,
            "commit": record.get("commit"),
            "instr_type": record.get("instr_type"),
            "status": "pending",
            "decision": None,
            "check": None,
            "semantic_attempts": [],
        }
        for line_number, record in enumerate(records, start=1)
    ]


def _load_state(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    state = read_jsonl(manifest["state_path"])
    if len(state) != manifest["input_count"]:
        raise InferenceError("Semantic state does not match the input record count")
    for line_number, item in enumerate(state, start=1):
        if item.get("line_number") != line_number:
            raise InferenceError("Semantic state is not in source order")
    return state


def _attempt_paths(run_dir: Path, attempt_number: int) -> dict[str, Path]:
    directory = run_dir / f"attempt_{attempt_number:03d}"
    return {
        "directory": directory,
        "prompt_path": directory / "prompts.jsonl",
        "inference_dir": directory / "inference",
        "raw_output_path": directory / "raw_responses.jsonl",
    }


def _sampling_overrides(manifest: dict[str, Any]) -> dict[str, Any]:
    overrides = dict(manifest["runtime_overrides"])
    overrides.update(
        {
            "num_completion": 1,
            "temperature": manifest["sampling"]["temperature"],
            "top_p": manifest["sampling"]["top_p"],
            "max_tokens": manifest["sampling"]["max_tokens"],
        }
    )
    response_format = {"response_format": {"type": "json_object"}}
    if manifest["executor"] == "api":
        overrides["extra_body"] = response_format
    else:
        overrides["request_body"] = response_format
    return overrides


def _start_attempt(
    manifest: dict[str, Any],
    state: list[dict[str, Any]],
    *,
    wait: bool,
    executor_instance: Any | None,
) -> int:
    attempt_number = len(manifest["attempts"]) + 1
    paths = _attempt_paths(Path(manifest["run_dir"]), attempt_number)
    pending_ids = {
        item["line_number"] for item in state if item["status"] == "pending"
    }
    if not pending_ids:
        return _finalize(manifest, state)

    _semantic_report(
        f"prepare attempt={attempt_number} samples={len(pending_ids)} "
        f"executor={manifest['executor']}"
    )
    system_prompt, user_prompt = get_prompts()
    _, all_prompts = _load_and_prepare_input(
        Path(manifest["input_path"]),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        pre_field=manifest["fields"]["pre"],
        post_field=manifest["fields"]["post"],
        instruction_field=manifest["fields"]["instruction"],
    )
    prompts = [item for item in all_prompts if item["prompt_id"] in pending_ids]
    paths["directory"].mkdir(parents=True, exist_ok=False)
    atomic_write_jsonl(paths["prompt_path"], prompts)
    attempt = {
        "attempt": attempt_number,
        "task_count": len(prompts),
        "prompt_path": str(paths["prompt_path"]),
        "inference_dir": str(paths["inference_dir"]),
        "raw_output_path": str(paths["raw_output_path"]),
        "status": "created",
    }
    manifest["attempts"].append(attempt)
    manifest["active_attempt"] = attempt_number
    manifest["status"] = "running"
    _save_manifest(manifest)

    _semantic_report(
        f"inference start attempt={attempt_number} samples={len(prompts)} "
        f"run_dir={paths['inference_dir']}"
    )
    result = create_run(
        executor=manifest["executor"],
        model=manifest["model_name"],
        config_path=manifest["config_path"],
        input_path=paths["prompt_path"],
        output_path=paths["raw_output_path"],
        run_dir=paths["inference_dir"],
        wait=wait,
        overrides=_sampling_overrides(manifest),
        executor_instance=executor_instance,
    )
    return _after_child_advance(
        manifest, result, wait=wait, executor_instance=executor_instance
    )


def _child_status(attempt: dict[str, Any]) -> dict[str, Any]:
    return show_status(attempt["inference_dir"])


def _raw_details(result_record: dict[str, Any]) -> tuple[str | None, dict[str, int]]:
    raw = result_record.get("raw_response")
    body = raw.get("body") if isinstance(raw, dict) and isinstance(raw.get("body"), dict) else raw
    finish_reason = None
    usage: dict[str, int] = {}
    if isinstance(body, dict):
        choices = body.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            value = choices[0].get("finish_reason")
            if isinstance(value, str):
                finish_reason = value
        raw_usage = body.get("usage")
        if isinstance(raw_usage, dict):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = raw_usage.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    usage[key] = value
    return finish_reason, usage


def _round_details(attempt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    manifest = load_yaml(Path(attempt["inference_dir"]) / "manifest.yaml")
    if manifest.get("round_count") != 1:
        raise InferenceError("Semantic inference attempts must contain one round")
    results_path = Path(manifest["rounds"][0]["results_path"])
    return {item["task_id"]: item for item in read_jsonl(results_path)}


def _process_attempt(
    manifest: dict[str, Any], state: list[dict[str, Any]], attempt: dict[str, Any]
) -> None:
    output_records = read_jsonl(attempt["raw_output_path"])
    details = _round_details(attempt)
    by_line = {item["line_number"]: item for item in state}
    seen: set[int] = set()
    valid_count = 0
    retry_count = 0
    exhausted_count = 0
    attempt_number = attempt["attempt"]
    maximum_attempts = manifest["semantic_retries"] + 1

    for output in output_records:
        line_number = output.get("source_line_number")
        if (
            isinstance(line_number, bool)
            or not isinstance(line_number, int)
            or line_number not in by_line
            or line_number in seen
        ):
            raise InferenceError(f"Invalid source_line_number in semantic output: {line_number!r}")
        seen.add(line_number)
        item = by_line[line_number]
        if item["status"] != "pending":
            previous = next(
                (
                    entry
                    for entry in item["semantic_attempts"]
                    if entry.get("attempt") == attempt_number
                ),
                None,
            )
            if previous is None:
                raise InferenceError(
                    f"Semantic output repeated resolved line {line_number}"
                )
            if item["status"] == "resolved":
                valid_count += 1
            else:
                exhausted_count += 1
            continue
        response = output.get("response_1")
        task_id = output.get("task_id")
        detail = details.get(task_id)
        if detail is None:
            raise InferenceError(
                f"Semantic output has no matching round result for task {task_id!r}"
            )
        finish_reason, usage = _raw_details(detail)
        error: dict[str, str] | None = None
        check = None
        try:
            check = validate_response(response, finish_reason)
        except ResponseValidationError as validation_error:
            error = {
                "type": type(validation_error).__name__,
                "message": str(validation_error),
            }

        history = {
            "attempt": attempt_number,
            "raw_response": response,
            "finish_reason": finish_reason,
            "error": error,
            "usage": usage,
        }
        existing = [
            entry
            for entry in item["semantic_attempts"]
            if entry.get("attempt") != attempt_number
        ]
        item["semantic_attempts"] = existing + [history]
        if check is not None:
            item["status"] = "resolved"
            item["check"] = check
            item["decision"] = decide(check)
            valid_count += 1
        elif attempt_number < maximum_attempts:
            retry_count += 1
        else:
            item["status"] = "error"
            item["decision"] = "ERROR"
            exhausted_count += 1

    if len(seen) != attempt["task_count"]:
        raise InferenceError(
            "Semantic raw output count does not match the attempt task count"
        )
    atomic_write_jsonl(manifest["state_path"], state)
    attempt.update(
        {
            "status": "validated",
            "valid_count": valid_count,
            "retry_count": retry_count,
            "exhausted_count": exhausted_count,
            "validated_at": _now(),
        }
    )
    _save_manifest(manifest)


def _after_child_advance(
    manifest: dict[str, Any],
    result: int,
    *,
    wait: bool,
    executor_instance: Any | None,
) -> int:
    attempt = manifest["attempts"][-1]
    child = _child_status(attempt)
    attempt["status"] = child["status"]
    if child["status"] != "complete":
        manifest["status"] = child["status"]
        _save_manifest(manifest)
        _semantic_report(
            f"inference pending attempt={attempt['attempt']} "
            f"status={child['status']}"
        )
        return result

    _semantic_report(
        f"inference complete attempt={attempt['attempt']} "
        f"samples={attempt['task_count']}"
    )
    state = _load_state(manifest)
    _semantic_report(
        f"validation start attempt={attempt['attempt']} "
        f"samples={attempt['task_count']}"
    )
    _process_attempt(manifest, state, attempt)
    _semantic_report(
        f"validation complete attempt={attempt['attempt']} "
        f"valid={attempt['valid_count']} retry={attempt['retry_count']} "
        f"exhausted={attempt['exhausted_count']}"
    )
    pending_count = sum(item["status"] == "pending" for item in state)
    if pending_count:
        _semantic_report(
            f"retry prepare next_attempt={attempt['attempt'] + 1} "
            f"samples={pending_count}"
        )
        return _start_attempt(
            manifest, state, wait=wait, executor_instance=executor_instance
        )
    return _finalize(manifest, state)


def _sum_usage(attempts: Iterable[dict[str, Any]]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for attempt in attempts:
        total.update(attempt.get("usage", {}))
    return dict(total)


def _result_record(
    manifest: dict[str, Any], item: dict[str, Any], inference_model: str
) -> dict[str, Any]:
    attempts = item["semantic_attempts"]
    latest = attempts[-1]
    return {
        "line_number": item["line_number"],
        "commit": item.get("commit"),
        "instr_type": item.get("instr_type"),
        "model_name": manifest["model_name"],
        "inference_model": inference_model,
        "executor": manifest["executor"],
        "semantic_attempt_count": len(attempts),
        "semantic_attempts": attempts,
        "decision": item["decision"],
        "check": item["check"],
        "raw_response": latest["raw_response"],
        "error": latest["error"] if item["decision"] == "ERROR" else None,
        "usage": _sum_usage(attempts),
    }


def _atomic_write_selected_lines(
    source_path: Path, destination_path: Path, accepted: set[int]
) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_name(f".{destination_path.name}.tmp")
    try:
        with source_path.open("rb") as source, temporary.open("wb") as destination:
            for line_number, line in enumerate(source, start=1):
                if line_number in accepted:
                    destination.write(line)
            destination.flush()
            os.fsync(destination.fileno())
        temporary.replace(destination_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _finalize(manifest: dict[str, Any], state: list[dict[str, Any]]) -> int:
    _semantic_report(f"finalize start samples={len(state)}")
    child_manifest = load_yaml(
        Path(manifest["attempts"][0]["inference_dir"]) / "manifest.yaml"
    )
    inference_model = child_manifest["config"]["executor_config"]["model"]
    results = [_result_record(manifest, item, inference_model) for item in state]
    observed_decisions = Counter(item["decision"] for item in results)
    decision_counts = {
        decision: observed_decisions.get(decision, 0)
        for decision in ("ACCEPT", "REJECT", "UNCERTAIN", "ERROR")
    }
    verdict_counts = {
        key: {
            verdict: Counter(
                item["check"][key]["verdict"]
                for item in results
                if item["check"] is not None
            ).get(verdict, 0)
            for verdict in ("PASS", "FAIL", "UNCERTAIN")
        }
        for key in CHECK_KEYS
    }
    token_usage: Counter[str] = Counter()
    for item in results:
        token_usage.update(item["usage"])
    accepted = {
        item["line_number"] for item in results if item["decision"] == "ACCEPT"
    }
    output_paths = manifest["output_paths"]
    _semantic_report(f"write results path={output_paths['results']}")
    atomic_write_jsonl(output_paths["results"], results)
    _semantic_report(
        f"write filtered path={output_paths['filtered']} samples={len(accepted)}"
    )
    _atomic_write_selected_lines(
        Path(manifest["input_path"]), Path(output_paths["filtered"]), accepted
    )
    retried = [item for item in results if item["semantic_attempt_count"] > 1]
    summary = {
        "schema_version": 1,
        "status": "complete",
        "total": len(results),
        "decision_counts": decision_counts,
        "verdict_counts": verdict_counts,
        "token_usage": dict(token_usage),
        "semantic_retry_count": sum(
            item["semantic_attempt_count"] - 1 for item in results
        ),
        "retried_samples": len(retried),
        "retry_exhausted": decision_counts["ERROR"],
        "semantic_retries": manifest["semantic_retries"],
        "semantic_attempts": len(manifest["attempts"]),
        "executor": manifest["executor"],
        "model_name": manifest["model_name"],
        "input_file": manifest["input_path"],
        "input_sha256": manifest["input_sha256"],
        "prompt_sha256": manifest["prompt_sha256"],
        "config_sha256": manifest["config_sha256"],
        "run_dir": manifest["run_dir"],
        "result_file": output_paths["results"],
        "filtered_file": output_paths["filtered"],
        "summary_file": output_paths["summary"],
    }
    _semantic_report(f"write summary path={output_paths['summary']}")
    atomic_write_yaml(output_paths["summary"], summary)
    manifest["status"] = "complete"
    manifest["completed_at"] = _now()
    manifest["summary"] = {
        "decision_counts": decision_counts,
        "semantic_retry_count": summary["semantic_retry_count"],
        "retried_samples": summary["retried_samples"],
        "retry_exhausted": summary["retry_exhausted"],
    }
    _save_manifest(manifest)
    counts = " ".join(
        f"{decision}={decision_counts[decision]}"
        for decision in ("ACCEPT", "REJECT", "UNCERTAIN", "ERROR")
    )
    _semantic_report(f"complete {counts}")
    return 0


def create_semantic_run(
    *,
    executor: str,
    model: str,
    config_path: str | Path,
    input_path: str | Path,
    run_dir: str | Path,
    result_path: str | Path | None = None,
    filtered_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    pre_field: str = "code_before_purify",
    post_field: str = "code_after_purify",
    instruction_field: str = "instruct_purify",
    semantic_retries: int = 2,
    temperature: float = 0,
    top_p: float = 1,
    max_tokens: int = 1200,
    max_batch_attempts: int | None = None,
    visible_devices: str | None = None,
    concurrency: int | None = None,
    tensor_parallel_size: int | None = None,
    thinking: bool | None = None,
    wait: bool = False,
    executor_instance: Any | None = None,
) -> int:
    """Validate input, create an outer run, and start its first attempt."""

    if executor not in EXECUTORS:
        raise InferenceError(f"Unsupported executor: {executor}")
    if isinstance(semantic_retries, bool) or semantic_retries < 0:
        raise InferenceError("semantic_retries must be a non-negative integer")
    input_path = Path(input_path).resolve()
    run_dir = Path(run_dir).resolve()
    config_path = Path(config_path).resolve()
    defaults = _default_output_paths(input_path)
    result_path = Path(result_path or defaults[0]).resolve()
    filtered_path = Path(filtered_path or defaults[1]).resolve()
    summary_path = Path(summary_path or defaults[2]).resolve()
    system_prompt, user_prompt = get_prompts()
    records, _ = _load_and_prepare_input(
        input_path,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        pre_field=pre_field,
        post_field=post_field,
        instruction_field=instruction_field,
    )
    config_sha256 = sha256_file(config_path)
    _validate_new_paths(
        input_path, run_dir, result_path, filtered_path, summary_path
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / STATE_NAME
    state = _initial_state(records)
    atomic_write_jsonl(state_path, state)
    runtime_overrides = {
        key: value
        for key, value in {
            "max_batch_attempts": max_batch_attempts,
            "visible_devices": visible_devices,
            "concurrency": concurrency,
            "tensor_parallel_size": tensor_parallel_size,
            "thinking": thinking,
        }.items()
        if value is not None
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "created",
        "created_at": _now(),
        "updated_at": _now(),
        "run_dir": str(run_dir),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "input_count": len(records),
        "prompt_sha256": _prompt_hash(system_prompt, user_prompt),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "executor": executor,
        "model_name": model,
        "fields": {
            "pre": pre_field,
            "post": post_field,
            "instruction": instruction_field,
        },
        "sampling": {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        },
        "runtime_overrides": runtime_overrides,
        "semantic_retries": semantic_retries,
        "state_path": str(state_path),
        "output_paths": {
            "results": str(result_path),
            "filtered": str(filtered_path),
            "summary": str(summary_path),
        },
        "attempts": [],
        "active_attempt": None,
    }
    _save_manifest(manifest)
    return _start_attempt(
        manifest, state, wait=wait, executor_instance=executor_instance
    )


def _advance_existing(
    run_dir: str | Path,
    command: str,
    *,
    wait: bool = False,
    executor_instance: Any | None = None,
) -> int:
    manifest = _load_manifest(run_dir)
    if manifest["status"] == "complete":
        return 0
    _validate_snapshots(manifest)
    if not manifest["attempts"]:
        return _start_attempt(
            manifest,
            _load_state(manifest),
            wait=wait,
            executor_instance=executor_instance,
        )
    attempt = manifest["attempts"][-1]
    child = _child_status(attempt)
    if child["status"] == "complete":
        return _after_child_advance(
            manifest, 0, wait=wait, executor_instance=executor_instance
        )
    if command == "resume":
        result = resume_run(
            attempt["inference_dir"], executor_instance=executor_instance
        )
    elif command == "continue":
        result = continue_run(
            attempt["inference_dir"],
            wait=wait,
            executor_instance=executor_instance,
        )
    elif command == "retry":
        result = retry_run(
            attempt["inference_dir"],
            wait=wait,
            executor_instance=executor_instance,
        )
    else:
        raise InferenceError(f"Unsupported semantic command: {command}")
    return _after_child_advance(
        manifest, result, wait=wait, executor_instance=executor_instance
    )


def resume_semantic_run(
    run_dir: str | Path, *, executor_instance: Any | None = None
) -> int:
    return _advance_existing(
        run_dir, "resume", executor_instance=executor_instance
    )


def continue_semantic_run(
    run_dir: str | Path,
    *,
    wait: bool = False,
    executor_instance: Any | None = None,
) -> int:
    return _advance_existing(
        run_dir,
        "continue",
        wait=wait,
        executor_instance=executor_instance,
    )


def retry_semantic_run(
    run_dir: str | Path,
    *,
    wait: bool = False,
    executor_instance: Any | None = None,
) -> int:
    return _advance_existing(
        run_dir, "retry", wait=wait, executor_instance=executor_instance
    )


def semantic_status(run_dir: str | Path) -> dict[str, Any]:
    """Return a read-only combined semantic and active inference status."""

    manifest = _load_manifest(run_dir)
    state = _load_state(manifest)
    counts = Counter(item["status"] for item in state)
    summary: dict[str, Any] = {
        "status": manifest["status"],
        "executor": manifest["executor"],
        "model_name": manifest["model_name"],
        "input_count": manifest["input_count"],
        "resolved": counts.get("resolved", 0),
        "pending": counts.get("pending", 0),
        "error": counts.get("error", 0),
        "semantic_retry_count": sum(
            max(0, len(item["semantic_attempts"]) - 1) for item in state
        ),
        "retried_samples": sum(
            len(item["semantic_attempts"]) > 1 for item in state
        ),
        "attempt_count": len(manifest["attempts"]),
        "active_attempt": manifest.get("active_attempt"),
        "output_paths": manifest["output_paths"],
    }
    if manifest["attempts"]:
        summary["inference"] = _child_status(manifest["attempts"][-1])
    return summary


def _add_runtime_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-batch-attempts", type=int)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--visible-devices")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--tensor-parallel-size", type=int)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--thinking", dest="thinking", action="store_true")
    thinking.add_argument("--no-thinking", dest="thinking", action="store_false")
    parser.set_defaults(thinking=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run resumable semantic checks through the inference framework."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Create and start a semantic run.")
    run.add_argument("--executor", required=True, choices=EXECUTORS)
    run.add_argument("--model", required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--result-file", type=Path)
    run.add_argument("--filtered-file", type=Path)
    run.add_argument("--summary-file", type=Path)
    run.add_argument("--pre-field", default="code_before_purify")
    run.add_argument("--post-field", default="code_after_purify")
    run.add_argument("--instruction-field", default="instruct_purify")
    run.add_argument("--semantic-retries", type=int, default=2)
    run.add_argument("--wait", action="store_true")
    _add_runtime_overrides(run)

    resume = subparsers.add_parser("resume", help="Resume API or local inference.")
    resume.add_argument("--run-dir", type=Path, required=True)
    continue_parser = subparsers.add_parser(
        "continue", help="Continue SiliconFlow Batch inference."
    )
    continue_parser.add_argument("--run-dir", type=Path, required=True)
    continue_parser.add_argument("--wait", action="store_true")
    retry = subparsers.add_parser(
        "retry", help="Grant the active inference attempt a new transport budget."
    )
    retry.add_argument("--run-dir", type=Path, required=True)
    retry.add_argument("--wait", action="store_true")
    status = subparsers.add_parser("status", help="Show semantic run status.")
    status.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return create_semantic_run(
                executor=args.executor,
                model=args.model,
                config_path=args.config,
                input_path=args.input,
                run_dir=args.run_dir,
                result_path=args.result_file,
                filtered_path=args.filtered_file,
                summary_path=args.summary_file,
                pre_field=args.pre_field,
                post_field=args.post_field,
                instruction_field=args.instruction_field,
                semantic_retries=args.semantic_retries,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                max_batch_attempts=args.max_batch_attempts,
                visible_devices=args.visible_devices,
                concurrency=args.concurrency,
                tensor_parallel_size=args.tensor_parallel_size,
                thinking=args.thinking,
                wait=args.wait,
            )
        if args.command == "resume":
            return resume_semantic_run(args.run_dir)
        if args.command == "continue":
            return continue_semantic_run(args.run_dir, wait=args.wait)
        if args.command == "retry":
            return retry_semantic_run(args.run_dir, wait=args.wait)
        print(json.dumps(semantic_status(args.run_dir), ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
