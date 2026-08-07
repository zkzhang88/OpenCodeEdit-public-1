#!/usr/bin/env python3
"""Semantically verify code-editing examples with a fresh LLM context."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import (
    ALL_COMPLETED,
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
import difflib
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from tqdm import tqdm
import yaml

try:
    from .prompts_for_check import get_prompts
except ImportError:  # Support ``python generation/semantic_check_api.py``.
    from prompts_for_check import get_prompts


DEFAULT_API_CONFIG = Path(__file__).resolve().parent / "api_config.yaml"
VERDICTS = {"PASS", "FAIL", "UNCERTAIN"}
TERMINAL_DECISIONS = {"ACCEPT", "REJECT", "UNCERTAIN"}
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
    """Raised when an LLM response does not match the required JSON schema."""


def load_model_config(api_config_path: Path, model_name: str) -> Dict[str, object]:
    """Load only the credentials and endpoint required for ``model_name``."""

    try:
        with api_config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"API config file not found: {api_config_path}. Copy "
            "api_config.example.yaml to api_config.yaml and fill in credentials."
        ) from exc

    if model_name == "qwen3-32b":
        names = ("QWEN_API_KEY", "QWEN_BASE_URL", "QWEN_API_MODEL_NAME")
        extra_body: Dict[str, object] = {"enable_thinking": False}
    elif model_name == "deepseek-v3":
        names = (
            "DEEPSEEK_API_KEY",
            "DEEPSEEK_BASE_URL",
            "DEEPSEEK_API_MODEL_NAME",
        )
        extra_body = {}
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    missing = [name for name in names if not config.get(name)]
    if missing:
        raise ValueError(
            f"Missing required fields in {api_config_path}: {', '.join(missing)}"
        )
    return {
        "api_key": config[names[0]],
        "base_url": config[names[1]],
        "api_model_name": config[names[2]],
        "extra_body": extra_body,
    }


def validate_prompt_template(user_prompt: str) -> None:
    """Require each supported placeholder exactly once."""

    invalid = [item for item in PLACEHOLDERS if user_prompt.count(item) != 1]
    if invalid:
        raise ValueError(
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
    """Render the fixed template without interpreting braces in input data."""

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


def _require_exact_keys(value: object, expected: set, location: str) -> dict:
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


def validate_response(raw_response: str, finish_reason: Optional[str] = None) -> dict:
    """Parse and validate one semantic-check response."""

    if finish_reason == "length":
        raise ResponseValidationError("Model response was truncated")
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise ResponseValidationError("Model returned an empty response")
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ResponseValidationError(
            f"Model response is not valid JSON: {exc}"
        ) from exc

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
        if key == "no_unrelated_changes":
            changes = check["unrelated_changes"]
            if not isinstance(changes, list):
                raise ResponseValidationError(
                    "no_unrelated_changes.unrelated_changes must be a list"
                )
    return parsed


def decide(check: dict) -> str:
    """Derive the filtering decision from the three independent verdicts."""

    verdicts = [check[key]["verdict"] for key in CHECK_KEYS]
    if all(verdict == "PASS" for verdict in verdicts):
        return "ACCEPT"
    if any(verdict == "FAIL" for verdict in verdicts):
        return "REJECT"
    return "UNCERTAIN"


def _usage_dict(usage: object) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, name, None) if usage is not None else None
        if isinstance(value, int) and not isinstance(value, bool):
            result[name] = value
    return result


def _add_usage(total: Dict[str, int], item: Dict[str, int]) -> None:
    for key, value in item.items():
        total[key] = total.get(key, 0) + value


def _record_metadata(record: object) -> Tuple[object, object]:
    if not isinstance(record, dict):
        return None, None
    return record.get("commit"), record.get("instr_type")


def _result_record(
    *,
    line_number: int,
    record: object,
    model_name: str,
    api_model_name: str,
    attempt: int,
    decision: str,
    check: Optional[dict],
    raw_response: Optional[str],
    error: Optional[dict],
    usage: Dict[str, int],
) -> dict:
    commit, instr_type = _record_metadata(record)
    return {
        "line_number": line_number,
        "commit": commit,
        "instr_type": instr_type,
        "model_name": model_name,
        "api_model_name": api_model_name,
        "attempt": attempt,
        "decision": decision,
        "check": check,
        "raw_response": raw_response,
        "error": error,
        "usage": usage,
    }


def _required_input(record: object, field: str) -> str:
    if not isinstance(record, dict):
        raise ValueError("Input line must contain a JSON object")
    if field not in record:
        raise ValueError(f"Input record is missing field '{field}'")
    value = record[field]
    if not isinstance(value, str):
        raise ValueError(f"Input field '{field}' must be a string")
    if not value.strip():
        raise ValueError(f"Input field '{field}' must not be empty")
    return value


def _check_delimiter_collision(
    instruction: str, pre_edit_code: str, post_edit_code: str, diff: str
) -> None:
    pairs = (
        (instruction, "</INSTRUCTION>"),
        (pre_edit_code, "</PRE_EDIT_CODE>"),
        (post_edit_code, "</POST_EDIT_CODE>"),
        (diff, "</DIFF>"),
    )
    collisions = [delimiter for value, delimiter in pairs if delimiter in value]
    if collisions:
        raise ValueError(
            "Input contains prompt closing delimiter(s): " + ", ".join(collisions)
        )


def check_one_record(
    *,
    line_number: int,
    raw_line: str,
    attempt: int,
    client: object,
    model_name: str,
    api_model_name: str,
    extra_body: dict,
    system_prompt: str,
    user_prompt_template: str,
    pre_field: str,
    post_field: str,
    instruction_field: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_retries: int,
) -> dict:
    """Validate and check one JSONL input line, never raising task errors."""

    try:
        record = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        return _result_record(
            line_number=line_number,
            record=None,
            model_name=model_name,
            api_model_name=api_model_name,
            attempt=attempt,
            decision="ERROR",
            check=None,
            raw_response=None,
            error={"type": "JSONDecodeError", "message": str(exc), "attempts": 0},
            usage={},
        )

    try:
        pre_edit_code = _required_input(record, pre_field)
        post_edit_code = _required_input(record, post_field)
        instruction = _required_input(record, instruction_field)
        diff = make_diff(pre_edit_code, post_edit_code)
        _check_delimiter_collision(instruction, pre_edit_code, post_edit_code, diff)
        user_prompt = render_user_prompt(
            user_prompt_template,
            instruction,
            pre_edit_code,
            post_edit_code,
            diff,
        )
    except (TypeError, ValueError) as exc:
        return _result_record(
            line_number=line_number,
            record=record,
            model_name=model_name,
            api_model_name=api_model_name,
            attempt=attempt,
            decision="ERROR",
            check=None,
            raw_response=None,
            error={"type": type(exc).__name__, "message": str(exc), "attempts": 0},
            usage={},
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    usage_total: Dict[str, int] = {}
    raw_response: Optional[str] = None
    last_error: Optional[Exception] = None
    attempts_made = 0

    for retry_index in range(max_retries + 1):
        attempts_made += 1
        try:
            arguments = {
                "model": api_model_name,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
            if extra_body:
                arguments["extra_body"] = extra_body
            completion = client.chat.completions.create(**arguments)
            if completion in (None, ""):
                raise ResponseValidationError("API returned an empty completion")
            choices = getattr(completion, "choices", None)
            if not choices:
                raise ResponseValidationError("API completion contains no choices")
            choice = choices[0]
            raw_response = getattr(getattr(choice, "message", None), "content", None)
            _add_usage(usage_total, _usage_dict(getattr(completion, "usage", None)))
            parsed = validate_response(
                raw_response,
                getattr(choice, "finish_reason", None),
            )
            return _result_record(
                line_number=line_number,
                record=record,
                model_name=model_name,
                api_model_name=api_model_name,
                attempt=attempt,
                decision=decide(parsed),
                check=parsed,
                raw_response=raw_response,
                error=None,
                usage=usage_total,
            )
        except ResponseValidationError as exc:
            # Reissue malformed outputs in a fresh context without rate-limit delay.
            last_error = exc
        except Exception as exc:  # API SDK exceptions vary by provider.
            last_error = exc
            if retry_index < max_retries:
                time.sleep(min(2**retry_index, 30))

    assert last_error is not None
    return _result_record(
        line_number=line_number,
        record=record,
        model_name=model_name,
        api_model_name=api_model_name,
        attempt=attempt,
        decision="ERROR",
        check=None,
        raw_response=raw_response,
        error={
            "type": type(last_error).__name__,
            "message": str(last_error),
            "attempts": attempts_made,
        },
        usage=usage_total,
    )


def _hash_file(path: Path) -> Tuple[str, int]:
    digest = hashlib.sha256()
    lines = 0
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            lines += 1
    return digest.hexdigest(), lines


def _prompt_hash(system_prompt: str, user_prompt: str) -> str:
    digest = hashlib.sha256()
    digest.update(system_prompt.encode("utf-8"))
    digest.update(b"\0")
    digest.update(user_prompt.encode("utf-8"))
    return digest.hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_state(path: Path, state: dict) -> None:
    _atomic_write_text(
        path, json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def _validate_progress_item(item: object, selected_total: int) -> dict:
    expected = {
        "line_number",
        "commit",
        "instr_type",
        "model_name",
        "api_model_name",
        "attempt",
        "decision",
        "check",
        "raw_response",
        "error",
        "usage",
    }
    if not isinstance(item, dict) or set(item) != expected:
        raise ValueError("Progress record has invalid fields")
    line_number = item["line_number"]
    attempt = item["attempt"]
    decision = item["decision"]
    if (
        isinstance(line_number, bool)
        or not isinstance(line_number, int)
        or not 1 <= line_number <= selected_total
        or decision not in TERMINAL_DECISIONS | {"ERROR"}
        or isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or attempt <= 0
    ):
        raise ValueError("Progress record has invalid identity or decision fields")
    if not isinstance(item["model_name"], str) or not isinstance(
        item["api_model_name"], str
    ):
        raise ValueError("Progress record has invalid model fields")
    if item["raw_response"] is not None and not isinstance(item["raw_response"], str):
        raise ValueError("Progress record has invalid raw_response")
    usage = item["usage"]
    if not isinstance(usage, dict) or any(
        not isinstance(key, str)
        or isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for key, value in usage.items()
    ):
        raise ValueError("Progress record has invalid token usage")
    if decision == "ERROR":
        if item["check"] is not None or not isinstance(item["error"], dict):
            raise ValueError("ERROR progress record must contain an error only")
    else:
        if item["error"] is not None:
            raise ValueError("Terminal progress record must not contain an error")
        try:
            check = validate_response(json.dumps(item["check"]))
        except ResponseValidationError as exc:
            raise ValueError(
                f"Progress record contains an invalid check: {exc}"
            ) from exc
        if decide(check) != decision:
            raise ValueError("Progress decision does not match its verdicts")
    return item


def _read_progress(
    path: Path, selected_total: int
) -> Tuple[Dict[int, dict], List[dict]]:
    """Read progress and repair only an incomplete final line."""

    if not path.exists() or path.stat().st_size == 0:
        return {}, []
    latest: Dict[int, dict] = {}
    history: List[dict] = []
    with path.open("r+b") as handle:
        file_size = os.fstat(handle.fileno()).st_size
        while True:
            line_start = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                break
            is_last = handle.tell() == file_size
            try:
                item = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if is_last:
                    handle.seek(line_start)
                    handle.truncate()
                    handle.flush()
                    os.fsync(handle.fileno())
                    break
                raise ValueError(
                    f"Invalid progress JSONL at byte offset {line_start}: {exc}"
                ) from exc
            try:
                item = _validate_progress_item(item, selected_total)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid progress record at byte offset {line_start}: {exc}"
                ) from exc
            line_number = item["line_number"]
            latest[line_number] = item
            history.append(item)
            if is_last and not raw_line.endswith(b"\n"):
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
    return latest, history


def _append_progress(handle, item: dict) -> None:
    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _selected_lines(input_file: Path, selected_total: int) -> Iterator[Tuple[int, str]]:
    with input_file.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if line_number > selected_total:
                break
            yield line_number, raw_line


def _drain_futures(
    pending: Dict[Future, int],
    progress_handle,
    latest: Dict[int, dict],
    progress,
    *,
    wait_for_all: bool,
) -> None:
    while pending:
        done, _ = wait(
            pending,
            return_when=ALL_COMPLETED if wait_for_all else FIRST_COMPLETED,
        )
        for future in done:
            line_number = pending.pop(future)
            item = future.result()
            _append_progress(progress_handle, item)
            latest[line_number] = item
            progress.update(1)
        if not wait_for_all:
            break


def _default_output(
    input_file: Path, suffix: str, extension: Optional[str] = None
) -> Path:
    return input_file.with_name(
        f"{input_file.stem}{suffix}{extension or input_file.suffix or '.jsonl'}"
    )


def _auxiliary_paths(result_file: Path) -> Tuple[Path, Path]:
    return (
        result_file.with_name(f"{result_file.stem}_progress.jsonl"),
        result_file.with_name(f"{result_file.stem}_state.json"),
    )


def _validate_output_paths(input_file: Path, paths: Iterable[Path]) -> None:
    resolved = [input_file.resolve()] + [path.resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError("Input, output, progress, and state paths must be distinct")


def _finalize_outputs(
    *,
    input_file: Path,
    selected_total: int,
    latest: Dict[int, dict],
    history: List[dict],
    result_file: Path,
    filtered_file: Path,
    summary_file: Path,
    state: dict,
) -> dict:
    missing = sorted(set(range(1, selected_total + 1)) - set(latest))
    if missing:
        raise RuntimeError(f"Semantic checking has {len(missing)} missing records")

    ordered = [latest[line_number] for line_number in range(1, selected_total + 1)]
    result_text = "".join(
        json.dumps(item, ensure_ascii=False) + "\n" for item in ordered
    )
    _atomic_write_text(result_file, result_text)

    accepted = {item["line_number"] for item in ordered if item["decision"] == "ACCEPT"}
    filtered_file.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=filtered_file.parent,
        prefix=f".{filtered_file.name}.",
        suffix=".tmp",
    )
    filtered_tmp = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as filtered_handle:
            for line_number, raw_line in _selected_lines(input_file, selected_total):
                if line_number in accepted:
                    filtered_handle.write(
                        raw_line if raw_line.endswith("\n") else raw_line + "\n"
                    )
            filtered_handle.flush()
            os.fsync(filtered_handle.fileno())
        os.replace(filtered_tmp, filtered_file)
    except Exception:
        try:
            filtered_tmp.unlink()
        except FileNotFoundError:
            pass
        raise

    decision_counts = Counter(item["decision"] for item in ordered)
    verdict_counts = {key: Counter() for key in CHECK_KEYS}
    for item in ordered:
        if item.get("check"):
            for key in CHECK_KEYS:
                verdict_counts[key][item["check"][key]["verdict"]] += 1
    token_usage: Counter = Counter()
    for item in history:
        token_usage.update(item.get("usage") or {})

    summary = {
        "input_file": str(input_file),
        "input_sha256": state["input_sha256"],
        "prompt_sha256": state["prompt_sha256"],
        "model_name": state["model_name"],
        "api_model_name": state["api_model_name"],
        "total": selected_total,
        "decision_counts": dict(sorted(decision_counts.items())),
        "verdict_counts": {
            key: dict(sorted(counts.items())) for key, counts in verdict_counts.items()
        },
        "token_usage": dict(sorted(token_usage.items())),
        "result_file": str(result_file),
        "filtered_file": str(filtered_file),
        "summary_file": str(summary_file),
    }
    _atomic_write_text(
        summary_file,
        yaml.safe_dump(summary, allow_unicode=True, sort_keys=False),
    )
    return summary


def run_semantic_check(
    *,
    input_file: Path,
    model_name: str,
    result_file: Optional[Path] = None,
    filtered_file: Optional[Path] = None,
    summary_file: Optional[Path] = None,
    api_config_file: Path = DEFAULT_API_CONFIG,
    pre_field: str = "code_before_purify",
    post_field: str = "code_after_purify",
    instruction_field: str = "instruct_purify",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 1200,
    max_samples: Optional[int] = None,
    max_retries: int = 5,
    workers: int = 1,
    continue_from_error: bool = False,
    show_progress: bool = True,
    client: Optional[object] = None,
    model_config: Optional[dict] = None,
) -> dict:
    """Run semantic checking and return the written summary."""

    input_file = Path(input_file)
    if model_name not in {"qwen3-32b", "deepseek-v3"}:
        raise ValueError(f"Unsupported model name: {model_name}")
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples cannot be negative")
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")

    result_file = Path(result_file or _default_output(input_file, "_semantic_results"))
    filtered_file = Path(
        filtered_file or _default_output(input_file, "_semantic_filtered")
    )
    summary_file = Path(
        summary_file or _default_output(input_file, "_semantic_summary", ".yaml")
    )
    progress_file, state_file = _auxiliary_paths(result_file)
    output_paths = (
        result_file,
        filtered_file,
        summary_file,
        progress_file,
        state_file,
    )
    _validate_output_paths(input_file, output_paths)
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    system_prompt, user_prompt_template = get_prompts()
    validate_prompt_template(user_prompt_template)
    input_sha256, total_lines = _hash_file(input_file)
    selected_total = (
        min(total_lines, max_samples) if max_samples is not None else total_lines
    )

    config = model_config or load_model_config(Path(api_config_file), model_name)
    api_model_name = str(config["api_model_name"])
    extra_body = dict(config.get("extra_body") or {})
    state = {
        "input_file": str(input_file.resolve()),
        "input_sha256": input_sha256,
        "prompt_sha256": _prompt_hash(system_prompt, user_prompt_template),
        "model_name": model_name,
        "api_model_name": api_model_name,
        "base_url": str(config.get("base_url", "")),
        "pre_field": pre_field,
        "post_field": post_field,
        "instruction_field": instruction_field,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "max_samples": max_samples,
    }

    if continue_from_error:
        if not state_file.is_file():
            raise ValueError(f"Cannot resume without state file: {state_file}")
        previous_state = json.loads(state_file.read_text(encoding="utf-8"))
        if previous_state != state:
            raise ValueError(
                "Current input, prompts, model, fields, or sampling settings "
                "differ from the saved run"
            )
    else:
        existing = [str(path) for path in output_paths if path.exists()]
        if existing:
            raise ValueError(
                "Refusing to overwrite existing semantic-check artifacts: "
                + ", ".join(existing)
            )
        _write_state(state_file, state)

    latest, history = _read_progress(progress_file, selected_total)
    if any(
        item["model_name"] != model_name or item["api_model_name"] != api_model_name
        for item in history
    ):
        raise ValueError("Progress records use a different configured model")
    terminal_lines = {
        line_number
        for line_number, item in latest.items()
        if item["decision"] in TERMINAL_DECISIONS
    }

    if client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The openai package is required for live semantic checking. "
                "Install generation/requirements.txt first."
            ) from exc
        client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])

    progress = tqdm(
        total=selected_total,
        initial=len(terminal_lines),
        desc="Semantic check",
        disable=not show_progress,
    )
    pending: Dict[Future, int] = {}
    with progress_file.open("a", encoding="utf-8") as progress_handle:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for line_number, raw_line in _selected_lines(input_file, selected_total):
                if line_number in terminal_lines:
                    continue
                previous = latest.get(line_number)
                attempt = int(previous.get("attempt", 0)) + 1 if previous else 1
                future = executor.submit(
                    check_one_record,
                    line_number=line_number,
                    raw_line=raw_line,
                    attempt=attempt,
                    client=client,
                    model_name=model_name,
                    api_model_name=api_model_name,
                    extra_body=extra_body,
                    system_prompt=system_prompt,
                    user_prompt_template=user_prompt_template,
                    pre_field=pre_field,
                    post_field=post_field,
                    instruction_field=instruction_field,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                )
                pending[future] = line_number
                if len(pending) >= workers * 2:
                    _drain_futures(
                        pending,
                        progress_handle,
                        latest,
                        progress,
                        wait_for_all=False,
                    )
            _drain_futures(
                pending,
                progress_handle,
                latest,
                progress,
                wait_for_all=True,
            )
    progress.close()

    # Reload progress so token usage includes previous attempts and resumed runs.
    latest, history = _read_progress(progress_file, selected_total)
    return _finalize_outputs(
        input_file=input_file,
        selected_total=selected_total,
        latest=latest,
        history=history,
        result_file=result_file,
        filtered_file=filtered_file,
        summary_file=summary_file,
        state=state,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Semantically check pre/post code edits with an LLM."
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument(
        "--model-name",
        required=True,
        choices=("qwen3-32b", "deepseek-v3"),
    )
    parser.add_argument("--result-file", type=Path, default=None)
    parser.add_argument("--filtered-file", type=Path, default=None)
    parser.add_argument("--summary-file", type=Path, default=None)
    parser.add_argument("--api-config-file", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--pre-field", default="code_before_purify")
    parser.add_argument("--post-field", default="code_after_purify")
    parser.add_argument("--instruction-field", default="instruct_purify")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--continue-from-error", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def print_summary(summary: dict) -> None:
    counts = summary["decision_counts"]
    print(
        f"Semantic check complete: total={summary['total']}, "
        f"accepted={counts.get('ACCEPT', 0)}, rejected={counts.get('REJECT', 0)}, "
        f"uncertain={counts.get('UNCERTAIN', 0)}, errors={counts.get('ERROR', 0)}"
    )
    print(f"Results: {summary['result_file']}")
    print(f"Filtered data: {summary['filtered_file']}")
    print(f"Summary YAML: {summary['summary_file']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_semantic_check(
            input_file=args.input_file,
            model_name=args.model_name,
            result_file=args.result_file,
            filtered_file=args.filtered_file,
            summary_file=args.summary_file,
            api_config_file=args.api_config_file,
            pre_field=args.pre_field,
            post_field=args.post_field,
            instruction_field=args.instruction_field,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            max_samples=args.max_samples,
            max_retries=args.max_retries,
            workers=args.workers,
            continue_from_error=args.continue_from_error,
            show_progress=not args.no_progress,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
