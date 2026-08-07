from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable

from .batch import build_batch_request, parse_batch_output, write_batch_inputs
from .config import load_credentials
from .io import InferenceError, append_jsonl, load_yaml, read_jsonl
from .schema import custom_id, parse_custom_id, round_messages


def _value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _object_id(value: Any) -> str:
    object_id = _value(value, "id")
    if object_id is None:
        data = _value(value, "data")
        object_id = _value(data, "id")
    if not isinstance(object_id, str) or not object_id:
        raise InferenceError(f"API response has no object ID: {value!r}")
    return object_id


def load_round_results(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return {}
    results: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or task_id in results:
            raise InferenceError(f"Invalid or duplicate task_id in {path}: {task_id!r}")
        results[task_id] = record
    return results


def save_successes(path: str | Path, successes: list[dict[str, Any]]) -> None:
    existing = load_round_results(path)
    for record in successes:
        task_id = record["task_id"]
        if task_id in existing:
            if existing[task_id]["content"] != record["content"]:
                raise InferenceError(f"Conflicting result for task {task_id}")
            continue
        append_jsonl(path, record)
        existing[task_id] = record


class RealtimeApiExecutor:
    def __init__(self, client_factory: Callable[..., Any] | None = None):
        self.client_factory = client_factory

    def advance_round(
        self,
        manifest: dict[str, Any],
        round_state: dict[str, Any],
        tasks: list[dict[str, Any]],
        prior_by_task: dict[str, list[str]],
        save_manifest: Callable[[], None],
        wait: bool,
    ) -> str:
        del wait
        config = manifest["config"]
        results_path = Path(round_state["results_path"])
        while True:
            completed = load_round_results(results_path)
            pending = [task for task in tasks if task["task_id"] not in completed]
            if not pending:
                round_state["status"] = "complete"
                save_manifest()
                return "complete"
            if round_state["attempts_used"] >= round_state["attempt_limit"]:
                round_state["status"] = "incomplete"
                save_manifest()
                return "incomplete"

            round_state["attempts_used"] += 1
            attempt = round_state["attempts_used"]
            attempt_path = Path(round_state["directory"]) / f"attempt_{attempt:03d}_output.jsonl"
            round_state["status"] = "running"
            save_manifest()

            api_key, base_url = load_credentials(config)
            if self.client_factory is None:
                from openai import OpenAI

                client_factory = OpenAI
            else:
                client_factory = self.client_factory
            client = client_factory(
                api_key=api_key,
                base_url=base_url,
                timeout=config["executor_config"].get("request_timeout", 600),
                max_retries=config["executor_config"].get("request_retries", 5),
            )
            successes: list[dict[str, Any]] = []
            for task in pending:
                try:
                    completion = client.chat.completions.create(
                        model=config["executor_config"]["model"],
                        messages=round_messages(
                            task, round_state["round"], prior_by_task[task["task_id"]]
                        ),
                        **config["sampling"],
                        extra_body=config["executor_config"].get("extra_body", {}),
                    )
                    content = completion.choices[0].message.content
                    if not isinstance(content, str) or not content.strip():
                        raise InferenceError("response has no non-empty assistant content")
                    raw_response = (
                        completion.model_dump(mode="json")
                        if hasattr(completion, "model_dump")
                        else None
                    )
                    attempt_record = {
                        "task_id": task["task_id"],
                        "round": round_state["round"],
                        "status": "success",
                        "content": content,
                        "raw_response": raw_response,
                    }
                    successes.append({key: value for key, value in attempt_record.items() if key != "status"})
                except Exception as error:  # one failed task must not discard the batch
                    attempt_record = {
                        "task_id": task["task_id"],
                        "round": round_state["round"],
                        "status": "error",
                        "error": f"{type(error).__name__}: {error}",
                    }
                append_jsonl(attempt_path, attempt_record)
            save_successes(results_path, successes)


class LlmInferBatchExecutor:
    def __init__(self, runner: Callable[..., Any] = subprocess.run):
        self.runner = runner

    @staticmethod
    def build_command(
        config: dict[str, Any], input_path: Path, output_path: Path, log_path: Path
    ) -> tuple[list[str], dict[str, str]]:
        executor_config = config["executor_config"]
        command = [
            str(executor_config.get("conda_executable", "conda")),
            "run",
            "--no-capture-output",
            "-n",
            str(executor_config.get("conda_environment", "llm_infer")),
            str(executor_config.get("command", "batch-infer")),
            "batch",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--model",
            str(executor_config["model"]),
            "--models-dir",
            str(executor_config.get("models_dir", "~/models")),
            "--concurrency",
            str(executor_config.get("concurrency", 8)),
            "--timeout",
            str(executor_config.get("timeout", 600)),
            "--max-retries",
            str(executor_config.get("max_retries", 3)),
            "--tensor-parallel-size",
            str(executor_config.get("tensor_parallel_size", 1)),
            "--gpu-memory-utilization",
            str(executor_config.get("gpu_memory_utilization", 0.9)),
            "--server-log",
            str(log_path),
            "--no-resume",
        ]
        if executor_config.get("auto_serve", True):
            command.append("--auto-serve")
        else:
            command.extend(["--no-auto-serve", "--base-url", str(executor_config["base_url"])])
        if executor_config.get("thinking") is True:
            command.append("--thinking")
        elif executor_config.get("thinking") is False:
            command.append("--no-thinking")
        if executor_config.get("max_model_len") is not None:
            command.extend(["--max-model-len", str(executor_config["max_model_len"])])
        if executor_config.get("chat_template"):
            command.extend(["--chat-template", str(executor_config["chat_template"])])
        if executor_config.get("trust_remote_code"):
            command.append("--trust-remote-code")

        environment = os.environ.copy()
        api_key_field = executor_config.get("api_key_field")
        if api_key_field:
            credentials = load_yaml(config["api_config_path"])
            if not credentials.get(api_key_field):
                raise InferenceError(
                    f"Missing required field in API config: {api_key_field}"
                )
            environment["OPENAI_API_KEY"] = str(credentials[api_key_field])
        if executor_config.get("visible_devices") is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(executor_config["visible_devices"])
        return command, environment

    def advance_round(
        self,
        manifest: dict[str, Any],
        round_state: dict[str, Any],
        tasks: list[dict[str, Any]],
        prior_by_task: dict[str, list[str]],
        save_manifest: Callable[[], None],
        wait: bool,
    ) -> str:
        del wait
        config = manifest["config"]
        results_path = Path(round_state["results_path"])
        while True:
            completed = load_round_results(results_path)
            pending = [task for task in tasks if task["task_id"] not in completed]
            if not pending:
                round_state["status"] = "complete"
                save_manifest()
                return "complete"
            if round_state["attempts_used"] >= round_state["attempt_limit"]:
                round_state["status"] = "incomplete"
                save_manifest()
                return "incomplete"

            round_state["attempts_used"] += 1
            attempt = round_state["attempts_used"]
            directory = Path(round_state["directory"])
            input_path = directory / f"attempt_{attempt:03d}_input.jsonl"
            output_path = directory / f"attempt_{attempt:03d}_output.jsonl"
            stdout_path = directory / f"attempt_{attempt:03d}.stdout.log"
            stderr_path = directory / f"attempt_{attempt:03d}.stderr.log"
            server_log = directory / f"attempt_{attempt:03d}.vllm.log"
            requests = [
                build_batch_request(
                    task,
                    round_state["round"],
                    prior_by_task[task["task_id"]],
                    config,
                )
                for task in pending
            ]
            write_batch_inputs(input_path, requests, max_records=None)
            command, environment = self.build_command(config, input_path, output_path, server_log)
            round_state["status"] = "running"
            round_state["last_command"] = command
            save_manifest()
            with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr_file:
                try:
                    completed_process = self.runner(
                        command,
                        shell=False,
                        env=environment,
                        stdout=stdout_file,
                        stderr=stderr_file,
                        check=False,
                    )
                except OSError as error:
                    stderr_file.write(f"{type(error).__name__}: {error}\n")
                    stderr_file.flush()
                    completed_process = type(
                        "FailedProcess", (), {"returncode": 127}
                    )()

            expected = {custom_id(task, round_state["round"]): task["task_id"] for task in pending}
            successes, failures = parse_batch_output(
                output_path, expected, round_state["round"]
            )
            save_successes(results_path, successes)
            round_state["last_exit_code"] = completed_process.returncode
            round_state["last_failures"] = failures
            save_manifest()


class SiliconFlowBatchExecutor:
    TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled", "canceled"}

    def __init__(
        self,
        client_factory: Callable[..., Any] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.client_factory = client_factory
        self.sleeper = sleeper

    def _client(self, config: dict[str, Any]) -> Any:
        api_key, base_url = load_credentials(config)
        if self.client_factory is None:
            from openai import OpenAI

            factory = OpenAI
        else:
            factory = self.client_factory
        return factory(api_key=api_key, base_url=base_url)

    @staticmethod
    def _write_remote_content(client: Any, file_id: str, path: Path) -> None:
        response = client.files.content(file_id)
        content = _value(response, "content")
        if isinstance(content, str):
            data = content.encode("utf-8")
        elif isinstance(content, bytes):
            data = content
        elif hasattr(response, "read"):
            data = response.read()
        elif hasattr(response, "write_to_file"):
            response.write_to_file(path)
            return
        else:
            raise InferenceError(f"Cannot read remote output file {file_id}")
        path.write_bytes(data)

    def _submit_attempt(
        self,
        client: Any,
        manifest: dict[str, Any],
        round_state: dict[str, Any],
        tasks: list[dict[str, Any]],
        prior_by_task: dict[str, list[str]],
        save_manifest: Callable[[], None],
    ) -> None:
        config = manifest["config"]
        executor_config = config["executor_config"]
        round_state["attempts_used"] += 1
        attempt = round_state["attempts_used"]
        directory = Path(round_state["directory"])
        completed = load_round_results(round_state["results_path"])
        pending = [task for task in tasks if task["task_id"] not in completed]
        requests = [
            build_batch_request(
                task,
                round_state["round"],
                prior_by_task[task["task_id"]],
                config,
            )
            for task in pending
        ]
        paths = write_batch_inputs(
            directory / f"attempt_{attempt:03d}_input.jsonl",
            requests,
            int(executor_config.get("max_requests_per_file", 6000)),
        )
        active = {"attempt": attempt, "parts": []}
        round_state["active_attempt"] = active
        round_state["status"] = "submitted"
        save_manifest()
        for part_number, input_path in enumerate(paths, start=1):
            with input_path.open("rb") as input_file:
                uploaded = client.files.create(file=input_file, purpose="batch")
            file_id = _object_id(uploaded)
            create_args: dict[str, Any] = {
                "input_file_id": file_id,
                "endpoint": executor_config.get("endpoint", "/v1/chat/completions"),
                "completion_window": executor_config.get("completion_window", "24h"),
                "metadata": {
                    "description": (
                        f"{Path(manifest['run_dir']).name} round {round_state['round']} "
                        f"attempt {attempt} part {part_number}"
                    )
                },
            }
            if executor_config.get("batch_extra_body") is not None:
                create_args["extra_body"] = executor_config["batch_extra_body"]
            job = client.batches.create(**create_args)
            active["parts"].append(
                {
                    "part": part_number,
                    "input_path": str(input_path),
                    "input_file_id": file_id,
                    "job_id": _object_id(job),
                    "status": str(_value(job, "status", "submitted")),
                }
            )
            save_manifest()

    def _poll_active(
        self,
        client: Any,
        round_state: dict[str, Any],
        save_manifest: Callable[[], None],
    ) -> bool:
        active = round_state["active_attempt"]
        all_terminal = True
        for part in active["parts"]:
            if part["status"] in self.TERMINAL_STATUSES:
                continue
            job = client.batches.retrieve(part["job_id"])
            status = str(_value(job, "status", "unknown")).lower()
            part["status"] = status
            output_file_id = _value(job, "output_file_id")
            if output_file_id:
                part["output_file_id"] = output_file_id
            error_file_id = _value(job, "error_file_id")
            if error_file_id:
                part["error_file_id"] = error_file_id
            if status not in self.TERMINAL_STATUSES:
                all_terminal = False
        save_manifest()
        return all_terminal

    def _collect_active(
        self,
        client: Any,
        round_state: dict[str, Any],
    ) -> dict[str, str]:
        active = round_state["active_attempt"]
        attempt = active["attempt"]
        failures: dict[str, str] = {}
        successes: list[dict[str, Any]] = []
        for part in active["parts"]:
            input_records = read_jsonl(part["input_path"])
            expected = {}
            for record in input_records:
                task_id, _ = parse_custom_id(record["custom_id"])
                expected[record["custom_id"]] = task_id
            output_path = Path(round_state["directory"]) / (
                f"attempt_{attempt:03d}_part{part['part']:03d}_output.jsonl"
            )
            if part.get("error_file_id"):
                error_path = Path(round_state["directory"]) / (
                    f"attempt_{attempt:03d}_part{part['part']:03d}_errors.jsonl"
                )
                self._write_remote_content(client, part["error_file_id"], error_path)
            if part["status"] == "completed" and part.get("output_file_id"):
                self._write_remote_content(client, part["output_file_id"], output_path)
                part_successes, part_failures = parse_batch_output(
                    output_path, expected, round_state["round"]
                )
                successes.extend(part_successes)
                failures.update(part_failures)
            else:
                reason = f"batch job ended with status {part['status']}"
                failures.update({task_id: reason for task_id in expected.values()})
        save_successes(round_state["results_path"], successes)
        return failures

    def advance_round(
        self,
        manifest: dict[str, Any],
        round_state: dict[str, Any],
        tasks: list[dict[str, Any]],
        prior_by_task: dict[str, list[str]],
        save_manifest: Callable[[], None],
        wait: bool,
    ) -> str:
        client = self._client(manifest["config"])
        poll_interval = float(
            manifest["config"]["executor_config"].get("poll_interval", 60)
        )
        while True:
            completed = load_round_results(round_state["results_path"])
            if len(completed) == len(tasks):
                round_state["status"] = "complete"
                round_state.pop("active_attempt", None)
                save_manifest()
                return "complete"

            active = round_state.get("active_attempt")
            if active is None:
                if round_state["attempts_used"] >= round_state["attempt_limit"]:
                    round_state["status"] = "incomplete"
                    save_manifest()
                    return "incomplete"
                self._submit_attempt(
                    client, manifest, round_state, tasks, prior_by_task, save_manifest
                )
                if not wait:
                    return "pending"
                active = round_state["active_attempt"]

            if not self._poll_active(client, round_state, save_manifest):
                if not wait:
                    return "pending"
                self.sleeper(poll_interval)
                continue

            failures = self._collect_active(client, round_state)
            round_state["last_failures"] = failures
            round_state.pop("active_attempt", None)
            save_manifest()
            if not wait and failures:
                if round_state["attempts_used"] >= round_state["attempt_limit"]:
                    round_state["status"] = "incomplete"
                    save_manifest()
                    return "incomplete"
                self._submit_attempt(
                    client, manifest, round_state, tasks, prior_by_task, save_manifest
                )
                return "pending"


def executor_for(name: str) -> Any:
    if name == "api":
        return RealtimeApiExecutor()
    if name == "llm-infer":
        return LlmInferBatchExecutor()
    if name == "siliconflow-batch":
        return SiliconFlowBatchExecutor()
    raise InferenceError(f"Unsupported executor: {name}")
