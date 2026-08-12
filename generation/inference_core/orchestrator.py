from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_credentials, resolve_config, validated_config_snapshot
from .executors import executor_for, load_round_results
from .io import (
    InferenceError,
    atomic_write_jsonl,
    atomic_write_yaml,
    load_yaml,
    sha256_file,
)
from .schema import expand_tasks, load_tasks, normalize_prompts, write_tasks


MANIFEST_NAME = "manifest.yaml"
TASKS_NAME = "tasks.jsonl"
MANIFEST_SCHEMA_VERSION = 2


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _manifest_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / MANIFEST_NAME


def _load_manifest(run_dir: str | Path) -> dict[str, Any]:
    manifest = load_yaml(_manifest_path(Path(run_dir).resolve()))
    version = manifest.get("schema_version")
    if version != MANIFEST_SCHEMA_VERSION:
        raise InferenceError(
            f"Unsupported manifest schema_version {version!r}; expected "
            f"{MANIFEST_SCHEMA_VERSION}. Create a new run with this version of "
            "the inference pipeline."
        )
    return manifest


def _save_manifest(manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _now()
    atomic_write_yaml(_manifest_path(manifest["run_dir"]), manifest)


def _validate_new_paths(run_dir: Path, output_path: Path) -> None:
    if output_path.exists():
        raise InferenceError(f"Output path already exists: {output_path}")
    if run_dir.exists():
        if not run_dir.is_dir():
            raise InferenceError(f"Run directory path is not a directory: {run_dir}")
        if any(run_dir.iterdir()):
            raise InferenceError(f"Run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)


def create_run(
    *,
    executor: str,
    model: str,
    config_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    run_dir: str | Path,
    wait: bool = False,
    overrides: dict[str, Any] | None = None,
    executor_instance: Any | None = None,
) -> int:
    config = resolve_config(config_path, executor, model, overrides)
    return create_run_from_config(
        executor=executor,
        model=model,
        config=config,
        input_path=input_path,
        output_path=output_path,
        run_dir=run_dir,
        wait=wait,
        executor_instance=executor_instance,
    )


def create_run_from_config(
    *,
    executor: str,
    model: str,
    config: dict[str, Any],
    input_path: str | Path,
    output_path: str | Path,
    run_dir: str | Path,
    wait: bool = False,
    executor_instance: Any | None = None,
) -> int:
    """Create a run from an already resolved, immutable config snapshot."""

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    run_dir = Path(run_dir).resolve()
    prompts = normalize_prompts(input_path)
    config = validated_config_snapshot(config, executor, model)
    tasks = expand_tasks(prompts, int(config["num_completion"]))
    round_count = max(len(task["prompt"]["user"]) for task in tasks)
    if executor in {"api", "siliconflow-batch"}:
        load_credentials(config)
    _validate_new_paths(run_dir, output_path)
    tasks_path = run_dir / TASKS_NAME
    write_tasks(tasks_path, tasks)

    rounds = []
    for round_index in range(1, round_count + 1):
        directory = run_dir / f"round_{round_index:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        rounds.append(
            {
                "round": round_index,
                "status": "pending",
                "directory": str(directory),
                "results_path": str(directory / "results.jsonl"),
                "expected_task_count": sum(
                    len(task["prompt"]["user"]) >= round_index for task in tasks
                ),
                "attempts_used": 0,
                "attempt_limit": int(config["max_batch_attempts"]),
            }
        )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "created",
        "created_at": _now(),
        "updated_at": _now(),
        "run_dir": str(run_dir),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output_path": str(output_path),
        "tasks_path": str(tasks_path),
        "task_count": len(tasks),
        "round_count": round_count,
        "executor": executor,
        "model_name": model,
        "config": config,
        "rounds": rounds,
    }
    _save_manifest(manifest)
    return _advance(manifest, wait=wait, executor_instance=executor_instance)


def _prior_responses(
    manifest: dict[str, Any], tasks: list[dict[str, Any]], round_index: int
) -> dict[str, list[str]]:
    prior = {task["task_id"]: [] for task in tasks}
    for previous_round in manifest["rounds"][: round_index - 1]:
        results = load_round_results(previous_round["results_path"])
        for task in tasks:
            if task["task_id"] not in results:
                raise InferenceError(
                    f"Round {previous_round['round']} has no result for task "
                    f"{task['task_id']} and cannot feed round {round_index}"
                )
            prior[task["task_id"]].append(results[task["task_id"]]["content"])
    return prior


def _final_records(
    manifest: dict[str, Any], tasks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    round_results = [load_round_results(item["results_path"]) for item in manifest["rounds"]]
    records = []
    for task in sorted(tasks, key=lambda item: (item["prompt_id"], item["sample_index"])):
        responses = [
            round_results[index][task["task_id"]]["content"]
            for index in range(len(task["prompt"]["user"]))
        ]
        output = dict(task["prompt"])
        for index, response in enumerate(responses, start=1):
            output[f"response_{index}"] = response
        output.update(
            {
                "response": responses,
                "sample_index": task["sample_index"],
                "task_id": task["task_id"],
                "model_name": manifest["model_name"],
                "executor": manifest["executor"],
            }
        )
        records.append(output)
    return records


def _finalize(manifest: dict[str, Any], tasks: list[dict[str, Any]]) -> None:
    output_path = Path(manifest["output_path"])
    if output_path.exists():
        raise InferenceError(f"Refusing to overwrite existing output: {output_path}")
    atomic_write_jsonl(output_path, _final_records(manifest, tasks))
    manifest["status"] = "complete"
    manifest["completed_at"] = _now()
    _save_manifest(manifest)


def _advance(
    manifest: dict[str, Any], wait: bool, executor_instance: Any | None = None
) -> int:
    tasks = load_tasks(manifest["tasks_path"])
    executor = executor_instance or executor_for(manifest["executor"])

    for round_state in manifest["rounds"]:
        if round_state["status"] == "complete":
            continue
        round_tasks = [
            task
            for task in tasks
            if len(task["prompt"]["user"]) >= round_state["round"]
        ]
        prior = _prior_responses(manifest, round_tasks, round_state["round"])

        def save() -> None:
            _save_manifest(manifest)

        try:
            result = executor.advance_round(
                manifest, round_state, round_tasks, prior, save, wait
            )
        except KeyboardInterrupt:
            round_state["status"] = "interrupted"
            round_state["last_exit_code"] = 130
            manifest["status"] = "interrupted"
            _save_manifest(manifest)
            return 130
        if result == "pending":
            manifest["status"] = "submitted"
            _save_manifest(manifest)
            return 0
        if result == "incomplete":
            manifest["status"] = "incomplete"
            _save_manifest(manifest)
            return 1
        if result == "interrupted":
            manifest["status"] = "interrupted"
            _save_manifest(manifest)
            return 130

    _finalize(manifest, tasks)
    return 0


def resume_run(
    run_dir: str | Path,
    *,
    executor_instance: Any | None = None,
) -> int:
    manifest = _load_manifest(run_dir)
    if manifest["executor"] == "siliconflow-batch":
        raise InferenceError(
            "SiliconFlow runs must be advanced with the 'continue' command"
        )
    if manifest["executor"] not in {"api", "llm-infer"}:
        raise InferenceError(
            f"The 'resume' command does not support executor {manifest['executor']!r}"
        )
    if manifest.get("status") == "complete":
        return 0
    _validate_input_snapshot(manifest)
    return _advance(manifest, wait=False, executor_instance=executor_instance)


def continue_run(
    run_dir: str | Path,
    *,
    wait: bool = False,
    executor_instance: Any | None = None,
) -> int:
    manifest = _load_manifest(run_dir)
    if manifest["executor"] != "siliconflow-batch":
        raise InferenceError(
            "Only SiliconFlow runs support the 'continue' command; use 'resume' "
            "for API and llm-infer runs"
        )
    if manifest.get("status") == "complete":
        return 0
    _validate_input_snapshot(manifest)
    return _advance(manifest, wait=wait, executor_instance=executor_instance)


def retry_run(
    run_dir: str | Path,
    *,
    wait: bool = False,
    executor_instance: Any | None = None,
) -> int:
    manifest = _load_manifest(run_dir)
    if manifest.get("status") != "incomplete":
        raise InferenceError("Only an incomplete run can receive a fresh retry budget")
    if any(round_state.get("active_attempt") for round_state in manifest["rounds"]):
        raise InferenceError(
            "Cannot retry while an attempt is active; resume or continue it first"
        )
    _validate_input_snapshot(manifest)
    budget = int(manifest["config"]["max_batch_attempts"])
    changed = False
    for round_state in manifest["rounds"]:
        if round_state["status"] == "incomplete":
            round_state["attempt_limit"] += budget
            round_state["status"] = "pending"
            changed = True
    if not changed:
        raise InferenceError("No incomplete round is available to retry")
    manifest["status"] = "running"
    _save_manifest(manifest)
    return _advance(manifest, wait=wait, executor_instance=executor_instance)


def _validate_input_snapshot(manifest: dict[str, Any]) -> None:
    if sha256_file(manifest["input_path"]) != manifest["input_sha256"]:
        raise InferenceError("Input file changed since the run was created")


def show_status(run_dir: str | Path) -> dict[str, Any]:
    manifest = _load_manifest(run_dir)
    summary = {
        "status": manifest["status"],
        "executor": manifest["executor"],
        "model_name": manifest["model_name"],
        "task_count": manifest["task_count"],
        "output_path": manifest["output_path"],
        "rounds": [],
    }
    for round_state in manifest["rounds"]:
        item = {
            "round": round_state["round"],
            "status": round_state["status"],
            "completed": len(load_round_results(round_state["results_path"])),
            "attempts_used": round_state["attempts_used"],
            "attempt_limit": round_state["attempt_limit"],
        }
        active = round_state.get("active_attempt")
        if active is not None:
            item["active_attempt"] = active.get("attempt")
            item["resume_count"] = active.get("resume_count", 0)
        elif "last_resume_count" in round_state:
            item["resume_count"] = round_state["last_resume_count"]
        if "last_exit_code" in round_state:
            item["last_exit_code"] = round_state["last_exit_code"]
        if "last_error" in round_state:
            item["last_error"] = round_state["last_error"]
        summary["rounds"].append(item)
    return summary
