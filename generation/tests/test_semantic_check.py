from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import yaml

from generation.inference_core.executors import (
    LlmInferBatchExecutor,
    RealtimeApiExecutor,
    SiliconFlowBatchExecutor,
)
from generation.inference_core.io import InferenceError
from generation.prompts_for_check import get_prompts
from generation.semantic_check import (
    ResponseValidationError,
    continue_semantic_run,
    create_semantic_run,
    main,
    make_diff,
    render_user_prompt,
    retry_semantic_run,
    resume_semantic_run,
    semantic_status,
    validate_prompt_template,
    validate_response,
)


def check_payload(
    pre: str = "PASS",
    post: str = "PASS",
    unrelated: str = "PASS",
    unrelated_changes: list[str] | None = None,
) -> dict:
    return {
        "pre_not_satisfied": {"verdict": pre, "reason": "pre reason"},
        "post_fulfills": {"verdict": post, "reason": "post reason"},
        "no_unrelated_changes": {
            "verdict": unrelated,
            "reason": "diff reason",
            "unrelated_changes": unrelated_changes or [],
        },
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class FakeCompletion:
    def __init__(self, content: str, finish_reason: str = "stop"):
        self.choices = [
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ]
        self._content = content
        self._finish_reason = finish_reason

    def model_dump(self, mode: str) -> dict:
        assert mode == "json"
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": self._content},
                    "finish_reason": self._finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }


class FakeApiCompletions:
    def __init__(self, responses: list[object]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        finish_reason = "stop"
        if isinstance(response, tuple):
            response, finish_reason = response
        if isinstance(response, dict):
            response = json.dumps(response)
        return FakeCompletion(response, finish_reason)


class FakeApiClient:
    def __init__(self, responses: list[object]):
        self.completions = FakeApiCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


class RecordingProgress:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.updates = []
        self.postfixes = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        del refresh
        values = dict(ordered_dict or {})
        values.update(kwargs)
        self.postfixes.append(values)

    def update(self, amount):
        self.updates.append(amount)

    def write(self, message, file=None):
        pass


class RecordingProgressFactory:
    def __init__(self):
        self.bars = []

    def __call__(self, **kwargs):
        progress = RecordingProgress(**kwargs)
        self.bars.append(progress)
        return progress


class FakeSiliconFlowFiles:
    def __init__(self, payload: dict):
        self.payload = payload
        self.inputs: dict[str, str] = {}
        self.request_bodies: list[dict] = []

    def create(self, file, purpose):
        assert purpose == "batch"
        file_id = f"file-{len(self.inputs) + 1}"
        self.inputs[file_id] = file.read().decode("utf-8")
        return SimpleNamespace(id=file_id)

    def content(self, output_file_id):
        input_file_id = output_file_id.removeprefix("output-")
        requests = [
            json.loads(line) for line in self.inputs[input_file_id].splitlines()
        ]
        results = []
        for request in requests:
            self.request_bodies.append(request["body"])
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
                                        "content": json.dumps(self.payload),
                                    },
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {"total_tokens": 12},
                        },
                    },
                    "error": None,
                }
            )
        content = "".join(json.dumps(item) + "\n" for item in results).encode()
        return SimpleNamespace(content=content)


class FakeSiliconFlowBatches:
    def __init__(self):
        self.jobs: dict[str, str] = {}

    def create(self, **kwargs):
        job_id = f"job-{len(self.jobs) + 1}"
        self.jobs[job_id] = kwargs["input_file_id"]
        return SimpleNamespace(id=job_id, status="in_progress")

    def retrieve(self, job_id):
        return SimpleNamespace(
            id=job_id,
            status="completed",
            output_file_id=f"output-{self.jobs[job_id]}",
            error_file_id=None,
        )


class FakeSiliconFlowClient:
    def __init__(self, payload: dict):
        self.files = FakeSiliconFlowFiles(payload)
        self.batches = FakeSiliconFlowBatches()


class PromptAndResponseTests(unittest.TestCase):
    def test_prompt_placeholders_braces_and_diff(self):
        system_prompt, user_prompt = get_prompts()
        self.assertIn("strict verifier", system_prompt)
        validate_prompt_template(user_prompt)
        rendered = render_user_prompt(
            user_prompt,
            "Change the mapping.",
            "value = {'before': 1}",
            "value = {'after': 2}",
            "-value = {'before': 1}\n+value = {'after': 2}",
        )
        self.assertIn("{'before': 1}", rendered)
        self.assertIn('"pre_not_satisfied"', rendered)
        self.assertNotIn("{instruction}", rendered)
        self.assertEqual(
            make_diff("value = 1\n", "value = 2\n").splitlines(),
            [
                "--- pre_edit.py",
                "+++ post_edit.py",
                "@@ -1 +1 @@",
                "-value = 1",
                "+value = 2",
            ],
        )

    def test_response_validation_is_strict(self):
        self.assertEqual(
            validate_response(json.dumps(check_payload()))["post_fulfills"]["verdict"],
            "PASS",
        )
        invalid_cases = (
            ("not json", None),
            (json.dumps(check_payload()), "length"),
            (json.dumps({**check_payload(), "extra": {}}), None),
            (json.dumps(check_payload(unrelated_changes="not a list")), None),
        )
        for response, finish_reason in invalid_cases:
            with self.subTest(response=response, finish_reason=finish_reason):
                with self.assertRaises(ResponseValidationError):
                    validate_response(response, finish_reason)


class SemanticWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_path = self.root / "static_filtered.jsonl"
        self.run_dir = self.root / "semantic_run"
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
                    "executors": {"siliconflow-batch": {"poll_interval": 0}},
                    "models": {
                        "test-api": {
                            "api": {
                                "model": "api-model",
                                "api_key_field": "TEST_KEY",
                                "base_url_field": "TEST_URL",
                                "extra_body": {"enable_thinking": False},
                            }
                        },
                        "test-sf": {
                            "siliconflow-batch": {"model": "remote-model"}
                        },
                        "test-local": {
                            "llm-infer": {
                                "model": "/models/local-model",
                                "models_dir": "/models",
                                "visible_devices": "0",
                                "tensor_parallel_size": 1,
                                "thinking": False,
                            }
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        self.records = [
            {
                "commit": f"commit-{index}",
                "instr_type": "qwen3_descriptive" if index % 2 else "ds_lazy",
                "code_before_purify": f"value = {index}\n",
                "code_after_purify": f"value = {index + 1}\n",
                "instruct_purify": "Increment value.",
            }
            for index in range(1, 4)
        ]
        write_jsonl(self.input_path, self.records)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def api_run(self, responses: list[object], progress_factory=None, **overrides):
        client = FakeApiClient(responses)
        executor = RealtimeApiExecutor(
            client_factory=lambda **kwargs: client,
            progress_factory=progress_factory,
        )
        arguments = {
            "executor": "api",
            "model": "test-api",
            "config_path": self.config_path,
            "input_path": self.input_path,
            "run_dir": self.run_dir,
            "executor_instance": executor,
        }
        arguments.update(overrides)
        result = create_semantic_run(**arguments)
        return result, client, executor

    def test_api_selective_retry_decisions_and_fresh_context(self):
        progress = RecordingProgressFactory()
        responses = [
            "not json",
            check_payload(post="FAIL"),
            check_payload(),
            check_payload(pre="UNCERTAIN"),
        ]
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result, client, _ = self.api_run(
                responses, progress_factory=progress
            )
        self.assertEqual(result, 0)
        report = stderr.getvalue()
        for expected in (
            "[semantic] prepare attempt=1 samples=3 executor=api",
            "[semantic] inference start attempt=1 samples=3",
            "[semantic] inference complete attempt=1 samples=3",
            "[semantic] validation start attempt=1 samples=3",
            "[semantic] validation complete attempt=1 valid=2 retry=1 exhausted=0",
            "[semantic] retry prepare next_attempt=2 samples=1",
            "[semantic] validation complete attempt=2 valid=1 retry=0 exhausted=0",
            "[semantic] finalize start samples=3",
            "[semantic] write results path=",
            "[semantic] write filtered path=",
            "[semantic] write summary path=",
            "[semantic] complete ACCEPT=1 REJECT=1 UNCERTAIN=1 ERROR=0",
        ):
            self.assertIn(expected, report)
        results_path = self.input_path.with_name(
            "static_filtered_semantic_results.jsonl"
        )
        results = read_jsonl(results_path)
        self.assertEqual(
            [item["decision"] for item in results],
            ["UNCERTAIN", "REJECT", "ACCEPT"],
        )
        self.assertEqual([item["semantic_attempt_count"] for item in results], [2, 1, 1])
        self.assertEqual(len(client.completions.calls), 4)
        for call in client.completions.calls:
            self.assertEqual([item["role"] for item in call["messages"]], ["system", "user"])
            self.assertEqual(call["temperature"], 0)
            self.assertEqual(call["top_p"], 1)
            self.assertEqual(call["max_tokens"], 1200)
            self.assertEqual(
                call["extra_body"],
                {
                    "enable_thinking": False,
                    "response_format": {"type": "json_object"},
                },
            )
        filtered = read_jsonl(
            self.input_path.with_name("static_filtered_semantic_filtered.jsonl")
        )
        self.assertEqual([item["commit"] for item in filtered], ["commit-3"])
        summary = yaml.safe_load(
            self.input_path.with_name("static_filtered_semantic_summary.yaml").read_text()
        )
        self.assertEqual(summary["semantic_retry_count"], 1)
        self.assertEqual(summary["retried_samples"], 1)
        self.assertEqual(summary["token_usage"]["total_tokens"], 60)
        self.assertEqual(semantic_status(self.run_dir)["status"], "complete")
        self.assertEqual([bar.options["total"] for bar in progress.bars], [3, 1])
        self.assertEqual([bar.options["initial"] for bar in progress.bars], [0, 0])
        self.assertEqual([bar.updates for bar in progress.bars], [[1, 1, 1], [1]])
        self.assertEqual(
            [bar.postfixes[-1] for bar in progress.bars],
            [{"success": 3, "failed": 0}, {"success": 1, "failed": 0}],
        )

    def test_invalid_response_retries_twice_then_becomes_error(self):
        write_jsonl(self.input_path, self.records[:1])
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result, client, _ = self.api_run(
                ["bad one", "bad two", (json.dumps(check_payload()), "length")]
            )
        self.assertEqual(result, 0)
        output = read_jsonl(
            self.input_path.with_name("static_filtered_semantic_results.jsonl")
        )[0]
        self.assertEqual(output["decision"], "ERROR")
        self.assertEqual(output["semantic_attempt_count"], 3)
        self.assertEqual(len(client.completions.calls), 3)
        self.assertEqual(output["error"]["message"], "Model response was truncated")
        self.assertIn(
            "[semantic] validation complete attempt=3 valid=0 retry=0 exhausted=1",
            stderr.getvalue(),
        )
        self.assertIn(
            "[semantic] complete ACCEPT=0 REJECT=0 UNCERTAIN=0 ERROR=1",
            stderr.getvalue(),
        )

    def test_invalid_input_fails_before_creating_run(self):
        invalid = dict(self.records[0])
        invalid["code_before_purify"] = "text = '</PRE_EDIT_CODE>'"
        write_jsonl(self.input_path, [invalid])
        client = FakeApiClient([])
        executor = RealtimeApiExecutor(client_factory=lambda **kwargs: client)
        with self.assertRaisesRegex(InferenceError, "closing delimiter"):
            create_semantic_run(
                executor="api",
                model="test-api",
                config_path=self.config_path,
                input_path=self.input_path,
                run_dir=self.run_dir,
                executor_instance=executor,
            )
        self.assertFalse(self.run_dir.exists())
        self.assertEqual(client.completions.calls, [])

    def test_api_resume_and_snapshot_validation(self):
        write_jsonl(self.input_path, self.records[:2])
        result, _, _ = self.api_run([check_payload(), KeyboardInterrupt()])
        self.assertEqual(result, 130)
        self.assertEqual(semantic_status(self.run_dir)["status"], "interrupted")

        changed = dict(self.records[0], instruct_purify="Different instruction.")
        write_jsonl(self.input_path, [changed, self.records[1]])
        with self.assertRaisesRegex(InferenceError, "Input file changed"):
            resume_semantic_run(self.run_dir)

        write_jsonl(self.input_path, self.records[:2])
        recovery_client = FakeApiClient([check_payload()])
        recovery = RealtimeApiExecutor(
            client_factory=lambda **kwargs: recovery_client
        )
        self.assertEqual(
            resume_semantic_run(self.run_dir, executor_instance=recovery), 0
        )
        self.assertEqual(len(recovery_client.completions.calls), 1)
        self.assertEqual(semantic_status(self.run_dir)["resolved"], 2)

    def test_transport_retry_budget_is_independent_from_semantic_retries(self):
        write_jsonl(self.input_path, self.records[:1])
        result, first_client, _ = self.api_run(
            [RuntimeError("transport one"), RuntimeError("transport two")]
        )
        self.assertEqual(result, 1)
        self.assertEqual(len(first_client.completions.calls), 2)
        self.assertEqual(semantic_status(self.run_dir)["status"], "incomplete")

        recovery_client = FakeApiClient([check_payload()])
        recovery = RealtimeApiExecutor(
            client_factory=lambda **kwargs: recovery_client
        )
        self.assertEqual(
            retry_semantic_run(self.run_dir, executor_instance=recovery), 0
        )
        output = read_jsonl(
            self.input_path.with_name("static_filtered_semantic_results.jsonl")
        )[0]
        self.assertEqual(output["decision"], "ACCEPT")
        self.assertEqual(output["semantic_attempt_count"], 1)
        self.assertEqual(len(recovery_client.completions.calls), 1)

    def test_siliconflow_submission_continue_and_json_mode(self):
        write_jsonl(self.input_path, self.records[:1])
        client = FakeSiliconFlowClient(check_payload())
        reports = []
        executor = SiliconFlowBatchExecutor(
            client_factory=lambda **kwargs: client,
            sleeper=lambda seconds: None,
            poll_reporter=reports.append,
        )
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = create_semantic_run(
                executor="siliconflow-batch",
                model="test-sf",
                config_path=self.config_path,
                input_path=self.input_path,
                run_dir=self.run_dir,
                executor_instance=executor,
            )
        self.assertEqual(result, 0)
        self.assertEqual(semantic_status(self.run_dir)["status"], "submitted")
        self.assertFalse(
            self.input_path.with_name("static_filtered_semantic_results.jsonl").exists()
        )
        with redirect_stderr(stderr):
            self.assertEqual(
                continue_semantic_run(
                    self.run_dir, wait=True, executor_instance=executor
                ),
                0,
            )
        self.assertEqual(semantic_status(self.run_dir)["status"], "complete")
        self.assertEqual(
            client.files.request_bodies[0]["response_format"],
            {"type": "json_object"},
        )
        self.assertIn("SiliconFlow poll #1", reports[0])
        self.assertIn("round=1 attempt=1", reports[0])
        self.assertIn("overall=completed", reports[0])
        self.assertIn("part=1 job_id=job-1", reports[1])
        self.assertIn("current_status=completed", reports[1])
        report_text = "\n".join(reports)
        self.assertIn("[siliconflow-batch] download start", report_text)
        self.assertIn("[siliconflow-batch] download complete", report_text)
        self.assertIn("[siliconflow-batch] parse complete", report_text)
        self.assertIn("[siliconflow-batch] merge complete", report_text)
        semantic_report = stderr.getvalue()
        self.assertIn("[semantic] inference pending attempt=1 status=submitted", semantic_report)
        self.assertIn("[semantic] validation start attempt=1 samples=1", semantic_report)
        self.assertIn(
            "[semantic] complete ACCEPT=1 REJECT=0 UNCERTAIN=0 ERROR=0",
            semantic_report,
        )

    def test_status_stdout_remains_json_and_stderr_is_empty(self):
        write_jsonl(self.input_path, self.records[:1])
        with redirect_stderr(io.StringIO()):
            result, _, _ = self.api_run([check_payload()])
        self.assertEqual(result, 0)

        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status_result = main(["status", "--run-dir", str(self.run_dir)])
        self.assertEqual(status_result, 0)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "complete")
        self.assertEqual(stderr.getvalue(), "")

    def test_local_batch_inference_uses_json_mode(self):
        write_jsonl(self.input_path, self.records[:1])
        request_bodies = []

        def runner(command, **kwargs):
            del kwargs
            input_path = Path(command[command.index("--input") + 1])
            output_path = Path(command[command.index("--output") + 1])
            requests = read_jsonl(input_path)
            results = []
            for request in requests:
                request_bodies.append(request["body"])
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
                                            "content": json.dumps(check_payload()),
                                        },
                                        "finish_reason": "stop",
                                    }
                                ]
                            },
                        },
                        "error": None,
                    }
                )
            write_jsonl(output_path, results)
            return SimpleNamespace(returncode=0)

        result = create_semantic_run(
            executor="llm-infer",
            model="test-local",
            config_path=self.config_path,
            input_path=self.input_path,
            run_dir=self.run_dir,
            executor_instance=LlmInferBatchExecutor(runner=runner),
        )
        self.assertEqual(result, 0)
        self.assertEqual(request_bodies[0]["response_format"], {"type": "json_object"})
        self.assertEqual(semantic_status(self.run_dir)["status"], "complete")


if __name__ == "__main__":
    unittest.main()
