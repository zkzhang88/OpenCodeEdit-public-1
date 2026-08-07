from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .io import InferenceError, atomic_write_jsonl, read_jsonl


CUSTOM_ID_PATTERN = re.compile(
    r"^task-([1-9]\d*)-([1-9]\d*)-round-([1-9]\d*)$"
)


def normalize_prompts(input_path: str | Path) -> list[dict[str, Any]]:
    records = read_jsonl(input_path)
    if not records:
        raise InferenceError(f"Prompt file is empty: {input_path}")

    prompt_ids: set[int] = set()
    normalized: list[dict[str, Any]] = []
    for line_number, record in enumerate(records, start=1):
        prompt_id = record.get("prompt_id")
        if isinstance(prompt_id, bool) or not isinstance(prompt_id, int) or prompt_id < 1:
            raise InferenceError(
                f"{input_path}:{line_number}: prompt_id must be a positive integer"
            )
        if prompt_id in prompt_ids:
            raise InferenceError(f"Duplicate prompt_id in input: {prompt_id}")
        prompt_ids.add(prompt_id)

        system = record.get("system", "You are a helpful assistant.")
        if not isinstance(system, str) or not system.strip():
            raise InferenceError(f"Prompt {prompt_id}: system must be non-empty")
        users = record.get("user")
        if isinstance(users, str):
            users = [users]
        if not isinstance(users, list) or not users or any(
            not isinstance(item, str) or not item.strip() for item in users
        ):
            raise InferenceError(
                f"Prompt {prompt_id}: user must be a non-empty string or list of strings"
            )

        normalized_record = dict(record)
        normalized_record["system"] = system
        normalized_record["user"] = list(users)
        normalized.append(normalized_record)
    return normalized


def expand_tasks(
    prompts: list[dict[str, Any]], num_completion: int
) -> list[dict[str, Any]]:
    if isinstance(num_completion, bool) or num_completion < 1:
        raise InferenceError("num_completion must be at least 1")
    return [
        {
            "task_id": f"{prompt['prompt_id']}:{sample_index}",
            "prompt_id": prompt["prompt_id"],
            "sample_index": sample_index,
            "prompt": prompt,
        }
        for prompt in prompts
        for sample_index in range(1, num_completion + 1)
    ]


def custom_id(task: dict[str, Any], round_index: int) -> str:
    return (
        f"task-{task['prompt_id']}-{task['sample_index']}"
        f"-round-{round_index}"
    )


def parse_custom_id(value: Any) -> tuple[str, int]:
    if not isinstance(value, str):
        raise InferenceError("custom_id must be a string")
    match = CUSTOM_ID_PATTERN.fullmatch(value)
    if match is None:
        raise InferenceError(f"Invalid custom_id: {value!r}")
    prompt_id, sample_index, round_index = map(int, match.groups())
    return f"{prompt_id}:{sample_index}", round_index


def load_tasks(path: str | Path) -> list[dict[str, Any]]:
    tasks = read_jsonl(path)
    task_ids: set[str] = set()
    for task in tasks:
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or task_id in task_ids:
            raise InferenceError(f"Invalid or duplicate task_id in {path}: {task_id!r}")
        task_ids.add(task_id)
    return tasks


def write_tasks(path: str | Path, tasks: list[dict[str, Any]]) -> None:
    atomic_write_jsonl(path, tasks)


def round_messages(
    task: dict[str, Any], round_index: int, prior_responses: list[str]
) -> list[dict[str, str]]:
    prompt = task["prompt"]
    if len(prior_responses) != round_index - 1:
        raise InferenceError(
            f"Task {task['task_id']} has {len(prior_responses)} prior responses "
            f"for round {round_index}"
        )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": prompt["system"]}
    ]
    for index in range(round_index):
        messages.append({"role": "user", "content": prompt["user"][index]})
        if index < len(prior_responses):
            messages.append(
                {"role": "assistant", "content": prior_responses[index]}
            )
    return messages
