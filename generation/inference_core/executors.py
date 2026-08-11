from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable
from urllib.parse import urlsplit
from urllib.request import urlopen

from tqdm import tqdm

from .batch import (
    build_batch_request,
    inspect_batch_output,
    parse_batch_output,
    write_batch_inputs,
)
from .config import load_credentials
from .io import (
    InferenceError,
    append_jsonl,
    atomic_write_jsonl,
    load_yaml,
    read_jsonl,
    repair_incomplete_jsonl_tail,
)
from .schema import parse_custom_id, round_messages


def _stderr_report(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    def __init__(
        self,
        client_factory: Callable[..., Any] | None = None,
        progress_factory: Callable[..., Any] | None = None,
    ):
        self.client_factory = client_factory
        self.progress_factory = progress_factory or tqdm

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
                active = round_state.get("active_attempt")
                if active is not None:
                    round_state["last_resume_count"] = active["resume_count"]
                    round_state["last_exit_code"] = 0
                    round_state.pop("last_error", None)
                round_state.pop("active_attempt", None)
                save_manifest()
                return "complete"

            active = round_state.get("active_attempt")
            if active is None:
                if round_state["attempts_used"] >= round_state["attempt_limit"]:
                    round_state["status"] = "incomplete"
                    save_manifest()
                    return "incomplete"
                round_state["attempts_used"] += 1
                attempt = round_state["attempts_used"]
                directory = Path(round_state["directory"])
                input_path = directory / f"attempt_{attempt:03d}_input.jsonl"
                output_path = directory / f"attempt_{attempt:03d}_output.jsonl"
                atomic_write_jsonl(
                    input_path,
                    [
                        {
                            "task_id": task["task_id"],
                            "round": round_state["round"],
                            "model": config["executor_config"]["model"],
                            "messages": round_messages(
                                task,
                                round_state["round"],
                                prior_by_task[task["task_id"]],
                            ),
                            "sampling": config["sampling"],
                            "extra_body": config["executor_config"].get(
                                "extra_body", {}
                            ),
                        }
                        for task in pending
                    ],
                )
                active = {
                    "attempt": attempt,
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "expected_task_ids": [task["task_id"] for task in pending],
                    "invocations": 0,
                    "resume_count": 0,
                }
                round_state["active_attempt"] = active
                round_state["status"] = "running"
                save_manifest()

            expected_ids = set(active["expected_task_ids"])
            records = self._load_attempt_records(
                active["output_path"], expected_ids, round_state["round"]
            )
            self._merge_api_successes(results_path, records)
            missing_ids = expected_ids - records.keys()
            if not missing_ids:
                round_state["last_failures"] = {
                    task_id: record["error"]
                    for task_id, record in records.items()
                    if record["status"] == "error"
                }
                round_state["last_exit_code"] = 0
                round_state["last_resume_count"] = active["resume_count"]
                round_state.pop("last_error", None)
                round_state.pop("active_attempt", None)
                round_state["status"] = "running"
                save_manifest()
                continue

            active["invocations"] += 1
            active["resume_count"] = active["invocations"] - 1
            round_state["status"] = "running"
            round_state.pop("last_error", None)
            save_manifest()
            task_by_id = {task["task_id"]: task for task in tasks}
            attempt_path = Path(active["output_path"])
            try:
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
                success_count = sum(
                    record["status"] == "success" for record in records.values()
                )
                failure_count = len(records) - success_count
                description = (
                    f"API round {round_state['round']} attempt {active['attempt']}"
                )
                with self.progress_factory(
                    total=len(expected_ids),
                    initial=len(records),
                    desc=description,
                    unit="request",
                    dynamic_ncols=True,
                    file=sys.stderr,
                ) as progress:
                    progress.set_postfix(
                        success=success_count, failed=failure_count, refresh=False
                    )
                    for task_id in active["expected_task_ids"]:
                        if task_id not in missing_ids:
                            continue
                        task = task_by_id[task_id]
                        try:
                            completion = client.chat.completions.create(
                                model=config["executor_config"]["model"],
                                messages=round_messages(
                                    task,
                                    round_state["round"],
                                    prior_by_task[task_id],
                                ),
                                **config["sampling"],
                                extra_body=config["executor_config"].get(
                                    "extra_body", {}
                                ),
                            )
                            content = completion.choices[0].message.content
                            if not isinstance(content, str) or not content.strip():
                                raise InferenceError(
                                    "response has no non-empty assistant content"
                                )
                            raw_response = (
                                completion.model_dump(mode="json")
                                if hasattr(completion, "model_dump")
                                else None
                            )
                            attempt_record = {
                                "task_id": task_id,
                                "round": round_state["round"],
                                "status": "success",
                                "content": content,
                                "raw_response": raw_response,
                            }
                            success_count += 1
                        except KeyboardInterrupt:
                            raise
                        except Exception as error:  # one task must not discard the attempt
                            error_message = f"{type(error).__name__}: {error}"
                            attempt_record = {
                                "task_id": task_id,
                                "round": round_state["round"],
                                "status": "error",
                                "error": error_message,
                            }
                            failure_count += 1
                            progress.set_postfix(
                                success=success_count,
                                failed=failure_count,
                                refresh=False,
                            )
                            progress.write(
                                f"{description} task {task_id} failed: {error_message}",
                                file=sys.stderr,
                            )
                        append_jsonl(attempt_path, attempt_record)
                        if attempt_record["status"] == "success":
                            self._merge_api_successes(
                                results_path, {task_id: attempt_record}
                            )
                        progress.set_postfix(
                            success=success_count,
                            failed=failure_count,
                            refresh=False,
                        )
                        progress.update(1)
            except KeyboardInterrupt:
                round_state["status"] = "interrupted"
                round_state["last_exit_code"] = 130
                save_manifest()
                return "interrupted"
            except Exception as error:
                round_state["status"] = "interrupted"
                round_state["last_exit_code"] = 1
                round_state["last_error"] = f"{type(error).__name__}: {error}"
                save_manifest()
                return "interrupted"

    @staticmethod
    def _load_attempt_records(
        path: str | Path, expected_ids: set[str], round_index: int
    ) -> dict[str, dict[str, Any]]:
        path = Path(path)
        if not path.exists():
            return {}
        records: dict[str, dict[str, Any]] = {}
        for record in read_jsonl(path):
            task_id = record.get("task_id")
            if task_id not in expected_ids:
                raise InferenceError(f"Unexpected API attempt task_id: {task_id!r}")
            if task_id in records:
                raise InferenceError(f"Duplicate API attempt task_id: {task_id}")
            if record.get("round") != round_index:
                raise InferenceError(
                    f"API attempt task {task_id} belongs to a different round"
                )
            if record.get("status") not in {"success", "error"}:
                raise InferenceError(f"Invalid API attempt status for task {task_id}")
            if record["status"] == "success":
                content = record.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise InferenceError(
                        f"API attempt task {task_id} has invalid content"
                    )
            elif not isinstance(record.get("error"), str):
                raise InferenceError(f"API attempt task {task_id} has invalid error")
            records[task_id] = record
        return records

    @staticmethod
    def _merge_api_successes(
        results_path: Path, records: dict[str, dict[str, Any]]
    ) -> None:
        save_successes(
            results_path,
            [
                {key: value for key, value in record.items() if key != "status"}
                for record in records.values()
                if record["status"] == "success"
            ],
        )


class LlmInferBatchExecutor:
    def __init__(self, runner: Callable[..., Any] = subprocess.run):
        self.runner = runner

    @staticmethod
    def build_command(
        config: dict[str, Any],
        input_path: Path,
        output_path: Path,
        log_path: Path,
        *,
        resume: bool = False,
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
            "--resume" if resume else "--no-resume",
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
                round_state.pop("active_attempt", None)
                save_manifest()
                return "complete"

            active = round_state.get("active_attempt")
            if active is None:
                if round_state["attempts_used"] >= round_state["attempt_limit"]:
                    round_state["status"] = "incomplete"
                    save_manifest()
                    return "incomplete"
                round_state["attempts_used"] += 1
                attempt = round_state["attempts_used"]
                directory = Path(round_state["directory"])
                input_path = directory / f"attempt_{attempt:03d}_input.jsonl"
                output_path = directory / f"attempt_{attempt:03d}_output.jsonl"
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
                active = {
                    "attempt": attempt,
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "expected": {
                        request["custom_id"]: task["task_id"]
                        for request, task in zip(requests, pending)
                    },
                    "invocations": 0,
                    "resume_count": 0,
                }
                round_state["active_attempt"] = active
                round_state["status"] = "running"
                save_manifest()

            input_path = Path(active["input_path"])
            output_path = Path(active["output_path"])
            expected = dict(active["expected"])
            repair_incomplete_jsonl_tail(output_path)
            successes, failures, seen = inspect_batch_output(
                output_path, expected, round_state["round"]
            )
            save_successes(results_path, successes)
            if seen == set(expected):
                round_state["last_failures"] = failures
                round_state["last_resume_count"] = active["resume_count"]
                round_state.pop("active_attempt", None)
                round_state["status"] = "running"
                save_manifest()
                continue

            invocation = int(active["invocations"])
            is_resume = invocation > 0
            attempt = int(active["attempt"])
            directory = Path(round_state["directory"])
            log_stem = f"attempt_{attempt:03d}"
            if is_resume:
                log_stem += f"_resume_{invocation:03d}"
            stdout_path = directory / f"{log_stem}.stdout.log"
            stderr_path = directory / f"{log_stem}.stderr.log"
            server_log = directory / f"{log_stem}.vllm.log"
            command, environment = self.build_command(
                config,
                input_path,
                output_path,
                server_log,
                resume=is_resume,
            )
            active["invocations"] = invocation + 1
            active["resume_count"] = invocation
            round_state["status"] = "running"
            round_state["last_command"] = command
            round_state.pop("last_error", None)
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
                except KeyboardInterrupt:
                    round_state["status"] = "interrupted"
                    round_state["last_exit_code"] = 130
                    save_manifest()
                    return "interrupted"
                except Exception as error:
                    stderr_file.write(f"{type(error).__name__}: {error}\n")
                    stderr_file.flush()
                    round_state["status"] = "interrupted"
                    round_state["last_exit_code"] = 1
                    round_state["last_error"] = f"{type(error).__name__}: {error}"
                    save_manifest()
                    return "interrupted"

            repair_incomplete_jsonl_tail(output_path)
            successes, failures, seen = inspect_batch_output(
                output_path, expected, round_state["round"]
            )
            save_successes(results_path, successes)
            round_state["last_exit_code"] = completed_process.returncode
            round_state["last_failures"] = failures
            if seen != set(expected):
                round_state["status"] = "interrupted"
                missing = set(expected) - seen
                round_state["last_error"] = (
                    f"attempt output is missing {len(missing)} terminal records"
                )
                save_manifest()
                return "interrupted"
            round_state.pop("last_error", None)
            round_state["last_resume_count"] = active["resume_count"]
            round_state.pop("active_attempt", None)
            round_state["status"] = "running"
            save_manifest()


class SiliconFlowBatchExecutor:
    TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled", "canceled"}

    def __init__(
        self,
        client_factory: Callable[..., Any] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        poll_reporter: Callable[[str], None] | None = None,
        url_opener: Callable[..., Any] | None = None,
    ):
        self.client_factory = client_factory
        self.sleeper = sleeper
        self.poll_reporter = poll_reporter or _stderr_report
        self.url_opener = url_opener or urlopen

    def _client(self, config: dict[str, Any]) -> Any:
        api_key, base_url = load_credentials(config)
        if self.client_factory is None:
            from openai import OpenAI

            factory = OpenAI
        else:
            factory = self.client_factory
        return factory(api_key=api_key, base_url=base_url)

    def _write_remote_content(
        self, client: Any, remote_file: str, path: Path
    ) -> None:
        scheme = urlsplit(remote_file).scheme.lower()
        if scheme in {"http", "https"}:
            self._write_url_content(remote_file, path)
            return

        response = client.files.content(remote_file)
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
            raise InferenceError(f"Cannot read remote output file {remote_file}")
        path.write_bytes(data)

    def _write_url_content(self, url: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".download.tmp",
                delete=False,
            ) as output_file:
                temporary_path = Path(output_file.name)
                with self.url_opener(url, timeout=600) as response:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output_file.write(chunk)
                output_file.flush()
                os.fsync(output_file.fileno())
            os.replace(temporary_path, path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass

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
        active = {"attempt": attempt, "poll_count": 0, "parts": []}
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
        active["poll_count"] = int(active.get("poll_count", 0)) + 1
        poll_count = active["poll_count"]
        timestamp = _utc_timestamp()
        all_terminal = True
        details: list[dict[str, Any]] = []
        for part in active["parts"]:
            previous_status = str(part.get("status", "unknown")).lower()
            queried = previous_status not in self.TERMINAL_STATUSES
            if queried:
                try:
                    job = client.batches.retrieve(part["job_id"])
                except Exception as error:
                    save_manifest()
                    error_message = f"{type(error).__name__}: {error}"
                    self.poll_reporter(
                        f"[{timestamp}] SiliconFlow poll #{poll_count} query failed: "
                        f"round={round_state['round']} attempt={active['attempt']} "
                        f"part={part['part']} job_id={part['job_id']} "
                        f"error={error_message}"
                    )
                    raise
                status = str(_value(job, "status", "unknown")).lower()
                part["status"] = status
                output_file_id = _value(job, "output_file_id")
                if output_file_id:
                    part["output_file_id"] = output_file_id
                error_file_id = _value(job, "error_file_id")
                if error_file_id:
                    part["error_file_id"] = error_file_id
            else:
                status = previous_status
            if status not in self.TERMINAL_STATUSES:
                all_terminal = False
            details.append(
                {
                    "part": part["part"],
                    "job_id": part["job_id"],
                    "previous_status": previous_status,
                    "current_status": status,
                    "queried": queried,
                    "output_file_id": part.get("output_file_id"),
                    "error_file_id": part.get("error_file_id"),
                }
            )
        save_manifest()
        statuses = Counter(detail["current_status"] for detail in details)
        if not all_terminal:
            overall = "running"
        elif all(status == "completed" for status in statuses):
            overall = "completed"
        else:
            overall = "terminal_with_errors"
        status_summary = ",".join(
            f"{status}:{count}" for status, count in sorted(statuses.items())
        )
        self.poll_reporter(
            f"[{timestamp}] SiliconFlow poll #{poll_count}: "
            f"round={round_state['round']} attempt={active['attempt']} "
            f"overall={overall} parts={len(details)} statuses={status_summary or '-'}"
        )
        for detail in details:
            self.poll_reporter(
                f"  part={detail['part']} job_id={detail['job_id']} "
                f"previous_status={detail['previous_status']} "
                f"current_status={detail['current_status']} "
                f"queried={'yes' if detail['queried'] else 'no'} "
                f"output_file_id={detail['output_file_id'] or '-'} "
                f"error_file_id={detail['error_file_id'] or '-'}"
            )
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
            round_state["last_poll_count"] = active.get("poll_count", 0)
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
