from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import yaml

from generation.inference_core.batch import build_batch_request, parse_batch_output
from generation.inference_core.config import resolve_config
from generation.inference_core.executors import (
    LlmInferBatchExecutor,
    RealtimeApiExecutor,
    SiliconFlowBatchExecutor,
)
from generation.inference_core.io import InferenceError, load_yaml
from generation.inference_core.orchestrator import (
    continue_run,
    create_run,
    resume_run,
    retry_run,
    show_status,
)
from generation.inference_core.schema import expand_tasks, normalize_prompts


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


class FakeApiCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = f"answer:{kwargs['messages'][-1]['content']}"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class FakeApiClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeApiCompletions())


class RecordingProgress:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.updates = []
        self.postfixes = []
        self.messages = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.closed = True

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        del refresh
        values = dict(ordered_dict or {})
        values.update(kwargs)
        self.postfixes.append(values)

    def update(self, amount):
        self.updates.append(amount)

    def write(self, message, file=None):
        self.messages.append((message, file))


class RecordingProgressFactory:
    def __init__(self):
        self.bars = []

    def __call__(self, **kwargs):
        progress = RecordingProgress(**kwargs)
        self.bars.append(progress)
        return progress


class FakeSiliconFlowFiles:
    def __init__(self):
        self.inputs = {}
        self.content_calls = []

    def create(self, file, purpose):
        assert purpose == "batch"
        file_id = f"file-{len(self.inputs) + 1}"
        self.inputs[file_id] = file.read().decode("utf-8")
        return SimpleNamespace(id=file_id)

    def output_bytes(self, input_file_id):
        requests = [
            json.loads(line) for line in self.inputs[input_file_id].splitlines()
        ]
        results = []
        for request in reversed(requests):
            results.append(
                {
                    "custom_id": request["custom_id"],
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": (
                                            "remote:"
                                            + request["body"]["messages"][-1]["content"]
                                        ),
                                    }
                                }
                            ]
                        },
                    },
                    "error": None,
                }
            )
        return "".join(json.dumps(item) + "\n" for item in results).encode()

    def content(self, output_file_id):
        self.content_calls.append(output_file_id)
        input_file_id = output_file_id.removeprefix("output-")
        return SimpleNamespace(content=self.output_bytes(input_file_id))


class FakeSiliconFlowBatches:
    def __init__(self):
        self.jobs = {}

    def create(self, **kwargs):
        job_id = f"job-{len(self.jobs) + 1}"
        self.jobs[job_id] = kwargs["input_file_id"]
        return SimpleNamespace(id=job_id, status="in_progress")

    def retrieve(self, job_id):
        file_id = self.jobs[job_id]
        return SimpleNamespace(
            id=job_id,
            status="completed",
            output_file_id=f"output-{file_id}",
            error_file_id=None,
        )


class SequencedSiliconFlowBatches(FakeSiliconFlowBatches):
    def __init__(self, status_sequences):
        super().__init__()
        self.status_sequences = status_sequences
        self.retrieve_counts = {}

    def retrieve(self, job_id):
        count = self.retrieve_counts.get(job_id, 0)
        self.retrieve_counts[job_id] = count + 1
        sequence = self.status_sequences[int(job_id.removeprefix("job-")) - 1]
        status = sequence[min(count, len(sequence) - 1)]
        if isinstance(status, Exception):
            raise status
        file_id = self.jobs[job_id]
        return SimpleNamespace(
            id=job_id,
            status=status,
            output_file_id=f"output-{file_id}" if status == "completed" else None,
            error_file_id=None,
        )


class UrlSiliconFlowBatches(FakeSiliconFlowBatches):
    def retrieve(self, job_id):
        file_id = self.jobs[job_id]
        return SimpleNamespace(
            id=job_id,
            status="completed",
            output_file_id=(
                f"https://downloads.invalid/{file_id}.jsonl?signature=top-secret"
            ),
            error_file_id=None,
        )


class FakeSiliconFlowClient:
    def __init__(self):
        self.files = FakeSiliconFlowFiles()
        self.batches = FakeSiliconFlowBatches()


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_path = self.root / "prompts.jsonl"
        self.output_path = self.root / "responses.jsonl"
        self.run_dir = self.root / "run"
        self.api_config = self.root / "api_config.yaml"
        self.api_config.write_text(
            yaml.safe_dump(
                {
                    "TEST_KEY": "secret",
                    "TEST_URL": "https://example.invalid/v1",
                    "SILICONFLOW_API_KEY": "secret",
                    "SILICONFLOW_BASE_URL": "https://example.invalid/v1",
                }
            ),
            encoding="utf-8",
        )
        self.config_path = self.root / "inference_config.yaml"
        self.config_path.write_text(
            yaml.safe_dump(
                {
                    "api_config_path": "api_config.yaml",
                    "defaults": {"max_batch_attempts": 2},
                    "executors": {
                        "siliconflow-batch": {"poll_interval": 0},
                    },
                    "models": {
                        "test-api": {
                            "api": {
                                "model": "api-model",
                                "api_key_field": "TEST_KEY",
                                "base_url_field": "TEST_URL",
                            }
                        },
                        "test-sf": {
                            "siliconflow-batch": {"model": "remote-model"}
                        },
                        "test-local": {
                            "llm-infer": {
                                "model": "/models/local-model",
                                "models_dir": "/models",
                                "base_url": "http://127.0.0.1:8001/v1",
                                "server_startup_timeout": 3600,
                                "vllm_executable": "/tools/vllm-no-custom-all-reduce",
                                "visible_devices": "2,3",
                                "tensor_parallel_size": 2,
                                "thinking": False,
                            }
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        self.prompts = [
            {
                "prompt_id": 2,
                "system": "system",
                "user": ["first-2", "second-2"],
                "commit": ["c2"],
            },
            {
                "prompt_id": 1,
                "system": "system",
                "user": ["first-1", "second-1"],
                "commit": ["c1"],
            },
        ]
        write_jsonl(self.input_path, self.prompts)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_prompt_validation_task_expansion_and_batch_identity(self):
        prompts = normalize_prompts(self.input_path)
        tasks = expand_tasks(prompts, 2)
        self.assertEqual(
            [task["task_id"] for task in tasks], ["2:1", "2:2", "1:1", "1:2"]
        )
        config = resolve_config(self.config_path, "api", "test-api")
        request = build_batch_request(tasks[0], 1, [], config)
        self.assertEqual(request["custom_id"], "task-2-1-round-1")
        self.assertEqual(
            [message["role"] for message in request["body"]["messages"]],
            ["system", "user"],
        )

    def test_api_executor_completes_two_rounds_and_sorts_final_output(self):
        fake_client = FakeApiClient()
        progress = RecordingProgressFactory()
        executor = RealtimeApiExecutor(
            client_factory=lambda **kwargs: fake_client,
            progress_factory=progress,
        )
        result = create_run(
            executor="api",
            model="test-api",
            config_path=self.config_path,
            input_path=self.input_path,
            output_path=self.output_path,
            run_dir=self.run_dir,
            executor_instance=executor,
        )
        self.assertEqual(result, 0)
        output = [json.loads(line) for line in self.output_path.read_text().splitlines()]
        self.assertEqual([record["task_id"] for record in output], ["1:1", "2:1"])
        self.assertEqual(output[0]["response"], ["answer:first-1", "answer:second-1"])
        self.assertEqual(output[0]["executor"], "api")
        self.assertEqual(show_status(self.run_dir)["status"], "complete")
        self.assertNotIn("secret", (self.run_dir / "manifest.yaml").read_text())
        self.assertEqual(len(progress.bars), 2)
        for round_index, bar in enumerate(progress.bars, start=1):
            self.assertEqual(bar.options["desc"], f"API round {round_index} attempt 1")
            self.assertEqual(bar.options["total"], 2)
            self.assertEqual(bar.options["initial"], 0)
            self.assertEqual(bar.updates, [1, 1])
            self.assertEqual(bar.postfixes[-1], {"success": 2, "failed": 0})
            self.assertTrue(bar.closed)

    def test_api_resume_reuses_attempt_and_only_requests_missing_tasks(self):
        class InterruptingCompletions:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 2:
                    raise KeyboardInterrupt
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="first:" + kwargs["messages"][-1]["content"]
                            )
                        )
                    ]
                )

        write_jsonl(
            self.input_path,
            [dict(record, user=record["user"][:1]) for record in self.prompts],
        )
        first_completions = InterruptingCompletions()
        first_client = SimpleNamespace(
            chat=SimpleNamespace(completions=first_completions)
        )
        first_progress = RecordingProgressFactory()
        result = create_run(
            executor="api",
            model="test-api",
            config_path=self.config_path,
            input_path=self.input_path,
            output_path=self.output_path,
            run_dir=self.run_dir,
            executor_instance=RealtimeApiExecutor(
                client_factory=lambda **kwargs: first_client,
                progress_factory=first_progress,
            ),
        )
        self.assertEqual(result, 130)
        status = show_status(self.run_dir)
        self.assertEqual(status["status"], "interrupted")
        self.assertEqual(status["rounds"][0]["active_attempt"], 1)
        self.assertEqual(status["rounds"][0]["attempts_used"], 1)
        self.assertEqual(first_progress.bars[0].updates, [1])
        self.assertTrue(first_progress.bars[0].closed)

        recovery_client = FakeApiClient()
        recovery_progress = RecordingProgressFactory()
        self.api_config.write_text(
            yaml.safe_dump(
                {
                    "TEST_KEY": "rotated-secret",
                    "TEST_URL": "https://rotated.invalid/v1",
                    "SILICONFLOW_API_KEY": "secret",
                    "SILICONFLOW_BASE_URL": "https://example.invalid/v1",
                }
            ),
            encoding="utf-8",
        )
        self.config_path.unlink()
        recovery_clients = []

        def recovery_factory(**kwargs):
            recovery_clients.append(kwargs)
            return recovery_client

        result = resume_run(
            self.run_dir,
            executor_instance=RealtimeApiExecutor(
                client_factory=recovery_factory,
                progress_factory=recovery_progress,
            ),
        )
        self.assertEqual(result, 0)
        self.assertEqual(recovery_clients[0]["api_key"], "rotated-secret")
        self.assertEqual(recovery_clients[0]["base_url"], "https://rotated.invalid/v1")
        self.assertEqual(len(recovery_client.chat.completions.calls), 1)
        self.assertEqual(
            recovery_client.chat.completions.calls[0]["messages"][-1]["content"],
            "first-1",
        )
        manifest = load_yaml(self.run_dir / "manifest.yaml")
        self.assertEqual(manifest["rounds"][0]["attempts_used"], 1)
        self.assertEqual(show_status(self.run_dir)["rounds"][0]["resume_count"], 1)
        self.assertEqual(
            len(
                (self.run_dir / "round_001" / "attempt_001_output.jsonl")
                .read_text()
                .splitlines()
            ),
            2,
        )
        self.assertEqual(len(recovery_progress.bars), 1)
        resumed_bar = recovery_progress.bars[0]
        self.assertEqual(resumed_bar.options["total"], 2)
        self.assertEqual(resumed_bar.options["initial"], 1)
        self.assertEqual(resumed_bar.updates, [1])
        self.assertEqual(resumed_bar.postfixes[-1], {"success": 2, "failed": 0})

    def test_api_progress_reports_failed_task_and_new_transport_attempt(self):
        class FailingOnceCompletions:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("temporary outage")
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
                )

        write_jsonl(self.input_path, [dict(self.prompts[0], user=["only-round"])])
        completions = FailingOnceCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        progress = RecordingProgressFactory()
        result = create_run(
            executor="api",
            model="test-api",
            config_path=self.config_path,
            input_path=self.input_path,
            output_path=self.output_path,
            run_dir=self.run_dir,
            executor_instance=RealtimeApiExecutor(
                client_factory=lambda **kwargs: client,
                progress_factory=progress,
            ),
        )

        self.assertEqual(result, 0)
        self.assertEqual(len(progress.bars), 2)
        self.assertEqual(
            progress.bars[0].postfixes[-1], {"success": 0, "failed": 1}
        )
        self.assertIn(
            "task 2:1 failed: RuntimeError: temporary outage",
            progress.bars[0].messages[0][0],
        )
        self.assertEqual(
            progress.bars[1].postfixes[-1], {"success": 1, "failed": 0}
        )

    def test_mixed_single_and_multi_round_prompts_are_supported(self):
        mixed = [self.prompts[0], dict(self.prompts[1], user=["only-one-round"])]
        write_jsonl(self.input_path, mixed)
        fake_client = FakeApiClient()
        result = create_run(
            executor="api",
            model="test-api",
            config_path=self.config_path,
            input_path=self.input_path,
            output_path=self.output_path,
            run_dir=self.run_dir,
            executor_instance=RealtimeApiExecutor(
                client_factory=lambda **kwargs: fake_client
            ),
        )
        self.assertEqual(result, 0)
        output = [json.loads(line) for line in self.output_path.read_text().splitlines()]
        self.assertEqual(len(output[0]["response"]), 1)
        self.assertNotIn("response_2", output[0])
        self.assertEqual(len(output[1]["response"]), 2)

    def test_siliconflow_submit_then_continue_waits_through_round_two(self):
        fake_client = FakeSiliconFlowClient()
        executor = SiliconFlowBatchExecutor(
            client_factory=lambda **kwargs: fake_client,
            sleeper=lambda seconds: None,
        )
        result = create_run(
            executor="siliconflow-batch",
            model="test-sf",
            config_path=self.config_path,
            input_path=self.input_path,
            output_path=self.output_path,
            run_dir=self.run_dir,
            wait=False,
            executor_instance=executor,
        )
        self.assertEqual(result, 0)
        self.assertFalse(self.output_path.exists())
        status = show_status(self.run_dir)
        self.assertEqual(status["status"], "submitted")
        self.assertEqual(status["rounds"][0]["status"], "submitted")
        with self.assertRaisesRegex(InferenceError, "continue"):
            resume_run(self.run_dir, executor_instance=executor)

        self.config_path.unlink()
        result = continue_run(
            self.run_dir, wait=True, executor_instance=executor
        )
        self.assertEqual(result, 0)
        output = [json.loads(line) for line in self.output_path.read_text().splitlines()]
        self.assertEqual([record["task_id"] for record in output], ["1:1", "2:1"])
        self.assertEqual(output[0]["response"], ["remote:first-1", "remote:second-1"])
        self.assertEqual(len(fake_client.batches.jobs), 2)

    def test_siliconflow_downloads_url_output_without_files_api(self):
        config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        config["executors"]["siliconflow-batch"]["max_requests_per_file"] = 1
        self.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        fake_client = FakeSiliconFlowClient()
        fake_client.batches = UrlSiliconFlowBatches()
        downloads = []
        reports = []

        def open_url(url, timeout):
            downloads.append((url, timeout))
            file_name = url.split("?", 1)[0].rsplit("/", 1)[-1]
            file_id = file_name.removesuffix(".jsonl")
            return io.BytesIO(fake_client.files.output_bytes(file_id))

        executor = SiliconFlowBatchExecutor(
            client_factory=lambda **kwargs: fake_client,
            sleeper=lambda seconds: None,
            poll_reporter=reports.append,
            url_opener=open_url,
        )
        self.assertEqual(
            create_run(
                executor="siliconflow-batch",
                model="test-sf",
                config_path=self.config_path,
                input_path=self.input_path,
                output_path=self.output_path,
                run_dir=self.run_dir,
                wait=False,
                executor_instance=executor,
            ),
            0,
        )
        self.assertEqual(
            continue_run(self.run_dir, wait=True, executor_instance=executor),
            0,
        )

        output = [json.loads(line) for line in self.output_path.read_text().splitlines()]
        self.assertEqual([record["task_id"] for record in output], ["1:1", "2:1"])
        self.assertEqual(len(downloads), 4)
        self.assertTrue(
            all(url.startswith("https://downloads.invalid/") for url, _ in downloads)
        )
        self.assertTrue(all(timeout == 600 for _, timeout in downloads))
        self.assertEqual(fake_client.files.content_calls, [])
        report_text = "\n".join(reports)
        for expected in (
            "[siliconflow-batch] download start round=1 attempt=1 part=1",
            "[siliconflow-batch] download complete round=1 attempt=1 part=2",
            "[siliconflow-batch] parse start round=1 attempt=1 part=1",
            "[siliconflow-batch] parse complete round=1 attempt=1 part=2",
            "[siliconflow-batch] merge start round=1 attempt=1 parts=2",
            "[siliconflow-batch] merge complete round=1 attempt=1 completed=2 expected=2",
            "source=url",
            "bytes=",
        ):
            self.assertIn(expected, report_text)
        self.assertNotIn("top-secret", report_text)

    def test_siliconflow_reports_download_failure_without_leaking_url(self):
        reports = []

        def fail_download(url, timeout):
            del url, timeout
            raise RuntimeError("download unavailable")

        executor = SiliconFlowBatchExecutor(
            poll_reporter=reports.append,
            url_opener=fail_download,
        )
        round_state = {
            "round": 3,
            "directory": str(self.root),
            "active_attempt": {"attempt": 2},
        }
        part = {"part": 4}
        signed_url = "https://downloads.invalid/output.jsonl?signature=top-secret"
        with self.assertRaisesRegex(RuntimeError, "download unavailable"):
            executor._download_part_file(
                SimpleNamespace(),
                round_state,
                part,
                kind="output",
                remote_file=signed_url,
                path=self.root / "output.jsonl",
            )
        self.assertIn("download start round=3 attempt=2 part=4", reports[0])
        self.assertIn("source=url", reports[0])
        self.assertIn("download failed round=3 attempt=2 part=4", reports[1])
        self.assertIn("RuntimeError: download unavailable", reports[1])
        self.assertNotIn("top-secret", "\n".join(reports))

    def test_siliconflow_reports_parse_failure(self):
        input_path = self.root / "part_input.jsonl"
        write_jsonl(
            input_path,
            [{"custom_id": "task-1-1-round-1", "body": {}}],
        )
        reports = []
        executor = SiliconFlowBatchExecutor(poll_reporter=reports.append)
        client = SimpleNamespace(
            files=SimpleNamespace(
                content=lambda remote_file: SimpleNamespace(content=b"not-json\n")
            )
        )
        round_state = {
            "round": 1,
            "directory": str(self.root),
            "results_path": str(self.root / "round_results.jsonl"),
            "active_attempt": {
                "attempt": 1,
                "parts": [
                    {
                        "part": 1,
                        "input_path": str(input_path),
                        "status": "completed",
                        "output_file_id": "output-file-1",
                    }
                ],
            },
        }

        with self.assertRaisesRegex(InferenceError, "invalid JSON"):
            executor._collect_active(client, round_state)
        report_text = "\n".join(reports)
        self.assertIn("[siliconflow-batch] download complete", report_text)
        self.assertIn("[siliconflow-batch] parse start", report_text)
        self.assertIn("[siliconflow-batch] parse failed", report_text)
        self.assertNotIn("[siliconflow-batch] merge start", report_text)

    def test_siliconflow_continue_reports_each_poll_and_persists_count(self):
        write_jsonl(
            self.input_path,
            [dict(self.prompts[0], user=["only-round"])],
        )
        client = FakeSiliconFlowClient()
        client.batches = SequencedSiliconFlowBatches(
            [["in_progress", "completed"]]
        )
        reports = []
        executor = SiliconFlowBatchExecutor(
            client_factory=lambda **kwargs: client,
            sleeper=lambda seconds: None,
            poll_reporter=reports.append,
        )
        self.assertEqual(
            create_run(
                executor="siliconflow-batch",
                model="test-sf",
                config_path=self.config_path,
                input_path=self.input_path,
                output_path=self.output_path,
                run_dir=self.run_dir,
                wait=False,
                executor_instance=executor,
            ),
            0,
        )
        self.assertEqual(reports, [])

        self.assertEqual(
            continue_run(self.run_dir, wait=False, executor_instance=executor), 0
        )
        active = load_yaml(self.run_dir / "manifest.yaml")["rounds"][0][
            "active_attempt"
        ]
        self.assertEqual(active["poll_count"], 1)
        self.assertIn("SiliconFlow poll #1", reports[0])
        self.assertIn("overall=running", reports[0])
        self.assertIn("previous_status=in_progress", reports[1])
        self.assertIn("current_status=in_progress", reports[1])
        self.assertIn("queried=yes", reports[1])

        self.assertEqual(
            continue_run(self.run_dir, wait=True, executor_instance=executor), 0
        )
        self.assertIn("SiliconFlow poll #2", reports[2])
        self.assertIn("overall=completed", reports[2])
        self.assertIn("current_status=completed", reports[3])
        round_state = load_yaml(self.run_dir / "manifest.yaml")["rounds"][0]
        self.assertEqual(round_state["last_poll_count"], 2)

    def test_siliconflow_poll_reports_terminal_parts_without_querying_again(self):
        batches = SimpleNamespace()
        batches.calls = []

        def retrieve(job_id):
            batches.calls.append(job_id)
            status = "in_progress" if len(batches.calls) == 1 else "failed"
            return SimpleNamespace(
                id=job_id,
                status=status,
                output_file_id=None,
                error_file_id="error-running" if status == "failed" else None,
            )

        batches.retrieve = retrieve
        round_state = {
            "round": 2,
            "active_attempt": {
                "attempt": 3,
                "parts": [
                    {
                        "part": 1,
                        "job_id": "job-complete",
                        "status": "completed",
                        "output_file_id": "output-complete",
                    },
                    {"part": 2, "job_id": "job-running", "status": "in_progress"},
                ],
            },
        }
        reports = []
        executor = SiliconFlowBatchExecutor(poll_reporter=reports.append)

        self.assertFalse(
            executor._poll_active(
                SimpleNamespace(batches=batches), round_state, lambda: None
            )
        )
        self.assertEqual(batches.calls, ["job-running"])
        self.assertEqual(round_state["active_attempt"]["poll_count"], 1)
        self.assertIn("statuses=completed:1,in_progress:1", reports[0])
        self.assertIn("job_id=job-complete", reports[1])
        self.assertIn("queried=no", reports[1])
        self.assertIn("job_id=job-running", reports[2])
        self.assertIn("queried=yes", reports[2])

        self.assertTrue(
            executor._poll_active(
                SimpleNamespace(batches=batches), round_state, lambda: None
            )
        )
        self.assertEqual(batches.calls, ["job-running", "job-running"])
        self.assertEqual(round_state["active_attempt"]["poll_count"], 2)
        self.assertIn("SiliconFlow poll #2", reports[3])
        self.assertIn("overall=terminal_with_errors", reports[3])
        self.assertIn("queried=no", reports[4])
        self.assertIn("current_status=failed", reports[5])
        self.assertIn("error_file_id=error-running", reports[5])

    def test_siliconflow_poll_reports_query_errors(self):
        def retrieve(job_id):
            raise RuntimeError("temporary poll failure")

        round_state = {
            "round": 4,
            "active_attempt": {
                "attempt": 2,
                "parts": [
                    {"part": 7, "job_id": "job-error", "status": "in_progress"}
                ],
            },
        }
        saved_poll_counts = []
        reports = []
        executor = SiliconFlowBatchExecutor(poll_reporter=reports.append)

        with self.assertRaisesRegex(RuntimeError, "temporary poll failure"):
            executor._poll_active(
                SimpleNamespace(batches=SimpleNamespace(retrieve=retrieve)),
                round_state,
                lambda: saved_poll_counts.append(
                    round_state["active_attempt"]["poll_count"]
                ),
            )
        self.assertEqual(saved_poll_counts, [1])
        self.assertEqual(len(reports), 1)
        for expected in (
            "SiliconFlow poll #1 query failed",
            "round=4",
            "attempt=2",
            "part=7",
            "job_id=job-error",
            "RuntimeError: temporary poll failure",
        ):
            self.assertIn(expected, reports[0])

    def test_llm_infer_runs_each_round_with_safe_command(self):
        calls = []

        def runner(command, **kwargs):
            calls.append((command, kwargs))
            input_path = Path(command[command.index("--input") + 1])
            output_path = Path(command[command.index("--output") + 1])
            requests = [json.loads(line) for line in input_path.read_text().splitlines()]
            results = [
                {
                    "custom_id": request["custom_id"],
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": "local:" + request["body"]["messages"][-1]["content"],
                                    }
                                }
                            ]
                        },
                    },
                    "error": None,
                }
                for request in requests
            ]
            write_jsonl(output_path, results)
            return SimpleNamespace(returncode=0)

        executor = LlmInferBatchExecutor(runner=runner)
        result = create_run(
            executor="llm-infer",
            model="test-local",
            config_path=self.config_path,
            input_path=self.input_path,
            output_path=self.output_path,
            run_dir=self.run_dir,
            executor_instance=executor,
        )
        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 2)
        for command, kwargs in calls:
            self.assertEqual(
                command[:7],
                ["conda", "run", "--no-capture-output", "-n", "llm_infer", "batch-infer", "batch"],
            )
            self.assertIn("--auto-serve", command)
            self.assertEqual(
                command[command.index("--base-url") + 1],
                "http://127.0.0.1:8001/v1",
            )
            self.assertEqual(
                command[command.index("--server-startup-timeout") + 1], "3600"
            )
            self.assertEqual(
                command[command.index("--vllm-executable") + 1],
                "/tools/vllm-no-custom-all-reduce",
            )
            self.assertIn("--progress", command)
            self.assertIn("--no-thinking", command)
            self.assertFalse(kwargs["shell"])
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2,3")

    def test_llm_infer_streams_stderr_and_preserves_logs(self):
        progress = (
            b"Ensuring local vLLM.\n"
            b"\x1b[32mbatch inference\x1b[0m 1/2\r"
            b"batch complete: total=2 succeeded=2\n"
        )

        def runner(command, **kwargs):
            kwargs["stdout"].write("stdout-only\n")
            kwargs["stdout"].flush()
            os.write(kwargs["stderr"], progress[:24])
            os.write(kwargs["stderr"], progress[24:])
            input_path = Path(command[command.index("--input") + 1])
            output_path = Path(command[command.index("--output") + 1])
            requests = [json.loads(line) for line in input_path.read_text().splitlines()]
            write_jsonl(
                output_path,
                [
                    {
                        "custom_id": request["custom_id"],
                        "response": {
                            "status_code": 200,
                            "body": {
                                "choices": [
                                    {
                                        "message": {
                                            "role": "assistant",
                                            "content": "ok",
                                        }
                                    }
                                ]
                            },
                        },
                        "error": None,
                    }
                    for request in requests
                ],
            )
            return SimpleNamespace(returncode=0)

        write_jsonl(
            self.input_path,
            [dict(record, user=record["user"][:1]) for record in self.prompts],
        )
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = create_run(
                executor="llm-infer",
                model="test-local",
                config_path=self.config_path,
                input_path=self.input_path,
                output_path=self.output_path,
                run_dir=self.run_dir,
                executor_instance=LlmInferBatchExecutor(runner=runner),
            )
        self.assertEqual(result, 0)
        self.assertEqual(stderr.getvalue().encode(), progress)
        round_dir = self.run_dir / "round_001"
        self.assertEqual(
            (round_dir / "attempt_001.stderr.log").read_bytes(), progress
        )
        self.assertEqual(
            (round_dir / "attempt_001.stdout.log").read_text(encoding="utf-8"),
            "stdout-only\n",
        )
        self.assertNotIn("stdout-only", stderr.getvalue())

    def test_llm_infer_stderr_relay_isolates_console_and_log_failures(self):
        class BrokenConsole:
            def write(self, value):
                del value
                raise OSError("console closed")

            def flush(self):
                raise OSError("console closed")

        read_fd, write_fd = os.pipe()
        os.write(write_fd, b"progress survives\n")
        os.close(write_fd)
        log = io.BytesIO()
        relay_errors = []
        LlmInferBatchExecutor._relay_stderr(
            read_fd, log, BrokenConsole(), relay_errors
        )
        self.assertEqual(log.getvalue(), b"progress survives\n")
        self.assertEqual(relay_errors, [])

        class BrokenLog:
            def write(self, value):
                del value
                raise OSError("log unavailable")

            def flush(self):
                raise OSError("log unavailable")

        read_fd, write_fd = os.pipe()
        os.write(write_fd, b"progress remains visible\n")
        os.close(write_fd)
        console = io.StringIO()
        relay_errors = []
        LlmInferBatchExecutor._relay_stderr(
            read_fd, BrokenLog(), console, relay_errors
        )
        self.assertEqual(console.getvalue(), "progress remains visible\n")
        self.assertEqual(len(relay_errors), 1)
        self.assertIn("log unavailable", str(relay_errors[0]))

    def test_llm_infer_retries_only_failed_task(self):
        attempts = []

        def runner(command, **kwargs):
            del kwargs
            input_path = Path(command[command.index("--input") + 1])
            output_path = Path(command[command.index("--output") + 1])
            requests = [json.loads(line) for line in input_path.read_text().splitlines()]
            attempts.append([request["custom_id"] for request in requests])
            results = []
            for index, request in enumerate(requests):
                fail = len(attempts) == 1 and index == 0
                results.append(
                    {
                        "custom_id": request["custom_id"],
                        "response": None if fail else {
                            "status_code": 200,
                            "body": {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
                        },
                        "error": {"message": "failed"} if fail else None,
                    }
                )
            write_jsonl(output_path, results)
            return SimpleNamespace(returncode=0)

        single_round = [dict(record, user=record["user"][:1]) for record in self.prompts]
        write_jsonl(self.input_path, single_round)
        result = create_run(
            executor="llm-infer",
            model="test-local",
            config_path=self.config_path,
            input_path=self.input_path,
            output_path=self.output_path,
            run_dir=self.run_dir,
            executor_instance=LlmInferBatchExecutor(runner=runner),
        )
        self.assertEqual(result, 0)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(len(attempts[0]), 2)
        self.assertEqual(len(attempts[1]), 1)

    def test_llm_infer_resume_uses_same_files_and_skips_existing_output(self):
        calls = []

        def runner(command, **kwargs):
            calls.append(command)
            os.write(kwargs["stderr"], f"progress-{len(calls)}\n".encode())
            input_path = Path(command[command.index("--input") + 1])
            output_path = Path(command[command.index("--output") + 1])
            requests = [json.loads(line) for line in input_path.read_text().splitlines()]
            existing = (
                [json.loads(line) for line in output_path.read_text().splitlines()]
                if output_path.exists()
                else []
            )
            completed_ids = {record["custom_id"] for record in existing}
            remaining = [
                {
                    "custom_id": request["custom_id"],
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": "local:" + request["custom_id"],
                                    }
                                }
                            ]
                        },
                    },
                    "error": None,
                }
                for request in requests
                if request["custom_id"] not in completed_ids
            ]
            if not calls[:-1]:
                write_jsonl(output_path, remaining[:1])
                raise KeyboardInterrupt
            write_jsonl(output_path, existing + remaining)
            return SimpleNamespace(returncode=0)

        write_jsonl(
            self.input_path,
            [dict(record, user=record["user"][:1]) for record in self.prompts],
        )
        executor = LlmInferBatchExecutor(runner=runner)
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = create_run(
                executor="llm-infer",
                model="test-local",
                config_path=self.config_path,
                input_path=self.input_path,
                output_path=self.output_path,
                run_dir=self.run_dir,
                executor_instance=executor,
            )
        self.assertEqual(result, 130)
        manifest = load_yaml(self.run_dir / "manifest.yaml")
        active = manifest["rounds"][0]["active_attempt"]
        self.assertEqual(active["resume_count"], 0)
        self.assertIn("--no-resume", calls[0])

        self.config_path.unlink()
        with redirect_stderr(stderr):
            result = resume_run(self.run_dir, executor_instance=executor)
        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 2)
        self.assertIn("--resume", calls[1])
        for option in ("--input", "--output"):
            self.assertEqual(
                calls[0][calls[0].index(option) + 1],
                calls[1][calls[1].index(option) + 1],
            )
        manifest = load_yaml(self.run_dir / "manifest.yaml")
        self.assertEqual(manifest["rounds"][0]["attempts_used"], 1)
        status = show_status(self.run_dir)["rounds"][0]
        self.assertEqual(status["resume_count"], 1)
        self.assertEqual(status["last_exit_code"], 0)
        self.assertTrue(
            (self.run_dir / "round_001" / "attempt_001_resume_001.stdout.log").exists()
        )
        round_dir = self.run_dir / "round_001"
        self.assertEqual(
            (round_dir / "attempt_001.stderr.log").read_text(encoding="utf-8"),
            "progress-1\n",
        )
        self.assertEqual(
            (round_dir / "attempt_001_resume_001.stderr.log").read_text(
                encoding="utf-8"
            ),
            "progress-2\n",
        )
        self.assertIn("progress-1\n", stderr.getvalue())
        self.assertIn("progress-2\n", stderr.getvalue())

    def test_llm_infer_nonzero_partial_output_remains_resumable(self):
        call_count = 0

        def runner(command, **kwargs):
            nonlocal call_count
            del kwargs
            call_count += 1
            input_path = Path(command[command.index("--input") + 1])
            output_path = Path(command[command.index("--output") + 1])
            requests = [json.loads(line) for line in input_path.read_text().splitlines()]
            records = [
                {
                    "custom_id": request["custom_id"],
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [
                                {"message": {"role": "assistant", "content": "ok"}}
                            ]
                        },
                    },
                    "error": None,
                }
                for request in requests
            ]
            if call_count == 1:
                write_jsonl(output_path, records[:1])
                return SimpleNamespace(returncode=17)
            existing = [json.loads(line) for line in output_path.read_text().splitlines()]
            write_jsonl(output_path, existing + records[1:])
            return SimpleNamespace(returncode=0)

        write_jsonl(
            self.input_path,
            [dict(record, user=record["user"][:1]) for record in self.prompts],
        )
        executor = LlmInferBatchExecutor(runner=runner)
        result = create_run(
            executor="llm-infer",
            model="test-local",
            config_path=self.config_path,
            input_path=self.input_path,
            output_path=self.output_path,
            run_dir=self.run_dir,
            executor_instance=executor,
        )
        self.assertEqual(result, 130)
        status = show_status(self.run_dir)
        self.assertEqual(status["rounds"][0]["last_exit_code"], 17)
        manifest_path = self.run_dir / "manifest.yaml"
        manifest = load_yaml(manifest_path)
        manifest["status"] = "running"
        manifest["rounds"][0]["status"] = "running"
        manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
        self.assertEqual(resume_run(self.run_dir, executor_instance=executor), 0)

    def test_llm_infer_resume_skips_errors_then_retries_them_in_a_new_attempt(self):
        calls = []

        def runner(command, **kwargs):
            del kwargs
            calls.append(command)
            input_path = Path(command[command.index("--input") + 1])
            output_path = Path(command[command.index("--output") + 1])
            requests = [json.loads(line) for line in input_path.read_text().splitlines()]

            def success(request):
                return {
                    "custom_id": request["custom_id"],
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [
                                {"message": {"role": "assistant", "content": "ok"}}
                            ]
                        },
                    },
                    "error": None,
                }

            if len(calls) == 1:
                write_jsonl(
                    output_path,
                    [
                        {
                            "custom_id": requests[0]["custom_id"],
                            "response": None,
                            "error": {"message": "temporary"},
                        }
                    ],
                )
                return SimpleNamespace(returncode=12)
            if len(calls) == 2:
                existing = [
                    json.loads(line) for line in output_path.read_text().splitlines()
                ]
                write_jsonl(output_path, existing + [success(requests[1])])
                return SimpleNamespace(returncode=0)
            self.assertEqual(len(requests), 1)
            write_jsonl(output_path, [success(requests[0])])
            return SimpleNamespace(returncode=0)

        write_jsonl(
            self.input_path,
            [dict(record, user=record["user"][:1]) for record in self.prompts],
        )
        executor = LlmInferBatchExecutor(runner=runner)
        self.assertEqual(
            create_run(
                executor="llm-infer",
                model="test-local",
                config_path=self.config_path,
                input_path=self.input_path,
                output_path=self.output_path,
                run_dir=self.run_dir,
                executor_instance=executor,
            ),
            130,
        )
        self.assertEqual(resume_run(self.run_dir, executor_instance=executor), 0)
        self.assertEqual(len(calls), 3)
        self.assertIn("--resume", calls[1])
        self.assertTrue(
            calls[1][calls[1].index("--input") + 1].endswith(
                "attempt_001_input.jsonl"
            )
        )
        self.assertIn("--no-resume", calls[2])
        self.assertTrue(
            calls[2][calls[2].index("--input") + 1].endswith(
                "attempt_002_input.jsonl"
            )
        )

    def test_llm_infer_repairs_only_an_incomplete_final_json_fragment(self):
        call_count = 0

        def runner(command, **kwargs):
            nonlocal call_count
            del kwargs
            call_count += 1
            input_path = Path(command[command.index("--input") + 1])
            output_path = Path(command[command.index("--output") + 1])
            requests = [json.loads(line) for line in input_path.read_text().splitlines()]

            def success(request):
                return {
                    "custom_id": request["custom_id"],
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [
                                {"message": {"role": "assistant", "content": "ok"}}
                            ]
                        },
                    },
                    "error": None,
                }

            if call_count == 1:
                output_path.write_bytes(
                    (json.dumps(success(requests[0])) + "\n").encode()
                    + b'{"custom_id":'
                )
                return SimpleNamespace(returncode=9)
            existing = [json.loads(line) for line in output_path.read_text().splitlines()]
            write_jsonl(output_path, existing + [success(requests[1])])
            return SimpleNamespace(returncode=0)

        write_jsonl(
            self.input_path,
            [dict(record, user=record["user"][:1]) for record in self.prompts],
        )
        executor = LlmInferBatchExecutor(runner=runner)
        self.assertEqual(
            create_run(
                executor="llm-infer",
                model="test-local",
                config_path=self.config_path,
                input_path=self.input_path,
                output_path=self.output_path,
                run_dir=self.run_dir,
                executor_instance=executor,
            ),
            130,
        )
        self.assertTrue(
            (
                self.run_dir
                / "round_001"
                / "attempt_001_output.jsonl.corrupt_tail_001"
            ).exists()
        )
        self.assertEqual(resume_run(self.run_dir, executor_instance=executor), 0)

    def test_llm_infer_rejects_corruption_before_the_final_line(self):
        def runner(command, **kwargs):
            del kwargs
            input_path = Path(command[command.index("--input") + 1])
            output_path = Path(command[command.index("--output") + 1])
            requests = [json.loads(line) for line in input_path.read_text().splitlines()]
            valid = {
                "custom_id": requests[0]["custom_id"],
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [
                            {"message": {"role": "assistant", "content": "ok"}}
                        ]
                    },
                },
                "error": None,
            }
            output_path.write_text(
                json.dumps(valid) + "\n" + '{"broken":\n' + json.dumps(valid) + "\n",
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0)

        write_jsonl(
            self.input_path,
            [dict(record, user=record["user"][:1]) for record in self.prompts],
        )
        with self.assertRaisesRegex(InferenceError, "invalid JSON"):
            create_run(
                executor="llm-infer",
                model="test-local",
                config_path=self.config_path,
                input_path=self.input_path,
                output_path=self.output_path,
                run_dir=self.run_dir,
                executor_instance=LlmInferBatchExecutor(runner=runner),
            )

    def test_batch_parser_rejects_duplicate_custom_id(self):
        output = self.root / "batch.jsonl"
        record = {
            "custom_id": "task-1-1-round-1",
            "response": {
                "status_code": 200,
                "body": {"choices": [{"message": {"content": "ok"}}]},
            },
            "error": None,
        }
        write_jsonl(output, [record, record])
        with self.assertRaisesRegex(InferenceError, "Duplicate"):
            parse_batch_output(output, {record["custom_id"]: "1:1"}, 1)

    def test_incomplete_run_can_receive_a_fresh_retry_budget(self):
        should_succeed = False

        def runner(command, **kwargs):
            del kwargs
            input_path = Path(command[command.index("--input") + 1])
            output_path = Path(command[command.index("--output") + 1])
            requests = [json.loads(line) for line in input_path.read_text().splitlines()]
            records = []
            for request in requests:
                records.append(
                    {
                        "custom_id": request["custom_id"],
                        "response": (
                            {
                                "status_code": 200,
                                "body": {
                                    "choices": [
                                        {"message": {"role": "assistant", "content": "recovered"}}
                                    ]
                                },
                            }
                            if should_succeed
                            else None
                        ),
                        "error": None if should_succeed else {"message": "still unavailable"},
                    }
                )
            write_jsonl(output_path, records)
            return SimpleNamespace(returncode=0)

        write_jsonl(self.input_path, [dict(self.prompts[0], user=["only round"])])
        executor = LlmInferBatchExecutor(runner=runner)
        result = create_run(
            executor="llm-infer",
            model="test-local",
            config_path=self.config_path,
            input_path=self.input_path,
            output_path=self.output_path,
            run_dir=self.run_dir,
            executor_instance=executor,
        )
        self.assertEqual(result, 1)
        self.assertFalse(self.output_path.exists())
        self.assertEqual(show_status(self.run_dir)["status"], "incomplete")

        should_succeed = True
        self.config_path.unlink()
        result = retry_run(self.run_dir, executor_instance=executor)
        self.assertEqual(result, 0)
        self.assertTrue(self.output_path.exists())
        manifest = load_yaml(self.run_dir / "manifest.yaml")
        self.assertEqual(manifest["rounds"][0]["attempt_limit"], 4)

    def test_command_executor_boundaries_and_retry_state_are_enforced(self):
        write_jsonl(self.input_path, [dict(self.prompts[0], user=["only round"])])
        result = create_run(
            executor="api",
            model="test-api",
            config_path=self.config_path,
            input_path=self.input_path,
            output_path=self.output_path,
            run_dir=self.run_dir,
            executor_instance=RealtimeApiExecutor(
                client_factory=lambda **kwargs: FakeApiClient()
            ),
        )
        self.assertEqual(result, 0)
        with self.assertRaisesRegex(InferenceError, "Only SiliconFlow"):
            continue_run(self.run_dir)
        with self.assertRaisesRegex(InferenceError, "Only an incomplete run"):
            retry_run(self.run_dir)

    def test_old_manifest_schema_is_rejected_without_modification(self):
        self.run_dir.mkdir()
        manifest_path = self.run_dir / "manifest.yaml"
        manifest_path.write_text("schema_version: 1\n", encoding="utf-8")
        original = manifest_path.read_bytes()
        operations = (
            lambda: resume_run(self.run_dir),
            lambda: continue_run(self.run_dir),
            lambda: retry_run(self.run_dir),
            lambda: show_status(self.run_dir),
        )
        for operation in operations:
            with self.assertRaisesRegex(InferenceError, "Unsupported manifest"):
                operation()
            self.assertEqual(manifest_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
