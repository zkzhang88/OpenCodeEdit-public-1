from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io import InferenceError, atomic_write_jsonl, read_jsonl
from .schema import custom_id, parse_custom_id, round_messages


def build_batch_request(
    task: dict[str, Any],
    round_index: int,
    prior_responses: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    executor_config = config["executor_config"]
    body = {
        "model": executor_config["model"],
        "messages": round_messages(task, round_index, prior_responses),
        "stream": False,
        **config["sampling"],
    }
    extra_body = executor_config.get("request_body")
    if extra_body is not None:
        if not isinstance(extra_body, dict):
            raise InferenceError("executor request_body must be a mapping")
        body.update(extra_body)
    return {
        "custom_id": custom_id(task, round_index),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def write_batch_inputs(
    base_path: str | Path,
    requests: list[dict[str, Any]],
    max_records: int | None,
) -> list[Path]:
    base_path = Path(base_path)
    if not requests:
        raise InferenceError("Cannot create an empty batch input")
    if max_records is None or max_records <= 0 or len(requests) <= max_records:
        atomic_write_jsonl(base_path, requests)
        return [base_path]

    paths: list[Path] = []
    for index, start in enumerate(range(0, len(requests), max_records), start=1):
        path = base_path.with_name(
            f"{base_path.stem}_part{index:03d}{base_path.suffix}"
        )
        atomic_write_jsonl(path, requests[start : start + max_records])
        paths.append(path)
    return paths


def parse_batch_output(
    path: str | Path,
    expected: dict[str, str],
    round_index: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    successes, failures, seen = inspect_batch_output(path, expected, round_index)
    for value, task_id in expected.items():
        if value not in seen:
            failures[task_id] = "missing result"
    return successes, failures


def inspect_batch_output(
    path: str | Path,
    expected: dict[str, str],
    round_index: int,
) -> tuple[list[dict[str, Any]], dict[str, str], set[str]]:
    """Parse terminal records without classifying absent IDs as failures."""

    successes: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    seen: set[str] = set()
    if not Path(path).exists():
        return successes, failures, seen

    for record in read_jsonl(path):
        value = record.get("custom_id")
        task_id, parsed_round = parse_custom_id(value)
        if parsed_round != round_index:
            raise InferenceError(
                f"Output custom_id {value!r} belongs to round {parsed_round}, "
                f"not round {round_index}"
            )
        if value not in expected or expected[value] != task_id:
            raise InferenceError(f"Unexpected output custom_id: {value!r}")
        if value in seen:
            raise InferenceError(f"Duplicate output custom_id: {value}")
        seen.add(value)

        if record.get("error") is not None:
            failures[task_id] = json.dumps(record["error"], ensure_ascii=False)
            continue
        response = record.get("response")
        body = response.get("body") if isinstance(response, dict) else None
        status_code = response.get("status_code") if isinstance(response, dict) else None
        if status_code is not None and status_code != 200:
            failures[task_id] = f"HTTP status {status_code}"
            continue
        choices = body.get("choices") if isinstance(body, dict) else None
        message = choices[0].get("message") if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if (
            not isinstance(message, dict)
            or message.get("role", "assistant") != "assistant"
            or not isinstance(content, str)
            or not content.strip()
        ):
            failures[task_id] = "response has no non-empty assistant content"
            continue
        successes.append(
            {
                "task_id": task_id,
                "round": round_index,
                "content": content,
                "custom_id": value,
                "raw_response": response,
            }
        )

    return successes, failures, seen
