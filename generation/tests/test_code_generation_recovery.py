import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from generation import code_generation_api


class FakeCompletions:
    def __init__(self, fail_at=None):
        self.fail_at = fail_at
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_at == len(self.calls):
            raise RuntimeError("simulated interruption")
        message = SimpleNamespace(content=f"response-{len(self.calls)}")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, fail_at=None):
        self.completions = FakeCompletions(fail_at)
        self.chat = SimpleNamespace(completions=self.completions)


class ApiRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_file = self.root / "prompts.jsonl"
        self.output_file = self.root / "output.jsonl"
        self.records = [
            {
                "prompt_id": prompt_id,
                "system": "system",
                "user": f"prompt-{prompt_id}",
            }
            for prompt_id in range(1, 5)
        ]
        self.write_input(self.records)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_input(self, records):
        self.input_file.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    @staticmethod
    def read_jsonl(path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def run_api(self, client, **overrides):
        arguments = {
            "input_path": self.input_file,
            "output_path": self.output_file,
            "model_name": "qwen3-32b",
        }
        arguments.update(overrides)
        with mock.patch.object(code_generation_api, "OpenAI", return_value=client):
            code_generation_api.api_infer(**arguments)

    def test_resume_skips_completed_tasks_and_repairs_partial_tail(self):
        self.write_input(self.records[:2])
        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            self.run_api(FakeClient(fail_at=2), num_completion=2)

        self.assertEqual(
            [record["task_id"] for record in self.read_jsonl(self.output_file)],
            ["1:1"],
        )
        with open(self.output_file, "ab") as output_file:
            output_file.write(b'{"task_id":"partial')

        resumed_client = FakeClient()
        self.run_api(
            resumed_client,
            num_completion=2,
            continue_from_error=True,
        )

        task_ids = [record["task_id"] for record in self.read_jsonl(self.output_file)]
        self.assertEqual(task_ids, ["1:1", "1:2", "2:1", "2:2"])
        self.assertEqual(len(task_ids), len(set(task_ids)))
        self.assertEqual(len(resumed_client.completions.calls), 3)

    def test_resume_uses_first_max_samples_records(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            self.run_api(FakeClient(fail_at=2), max_samples=2)

        resumed_client = FakeClient()
        self.run_api(
            resumed_client,
            max_samples=2,
            continue_from_error=True,
        )

        expected_ids = [1, 2]
        output_ids = [record["prompt_id"] for record in self.read_jsonl(self.output_file)]
        self.assertEqual(output_ids, expected_ids)
        self.assertEqual(len(resumed_client.completions.calls), 1)

    def test_output_fields_always_keep_identity_fields(self):
        self.write_input(self.records[:1])
        self.run_api(FakeClient(), output_fields=["response"])

        output_record = self.read_jsonl(self.output_file)[0]
        self.assertEqual(output_record["response"], ["response-1"])
        for field in ("prompt_id", "sample_index", "task_id", "model_name"):
            self.assertIn(field, output_record)

    def test_invalid_input_fails_before_api_call(self):
        invalid_inputs = (
            [{"user": "missing id"}],
            [
                {"prompt_id": 1, "user": "one"},
                {"prompt_id": 1, "user": "duplicate"},
            ],
            [{"prompt_id": 1, "user": ""}],
        )
        for records in invalid_inputs:
            with self.subTest(records=records):
                self.write_input(records)
                client = FakeClient()
                with self.assertRaises(ValueError):
                    self.run_api(client)
                self.assertEqual(client.completions.calls, [])

    def test_output_validation_rejects_duplicate_foreign_and_wrong_model(self):
        expected_task_ids = {"1:1"}
        valid_record = {
            "prompt_id": 1,
            "sample_index": 1,
            "task_id": "1:1",
            "model_name": "qwen3-32b",
        }
        cases = (
            ([valid_record, valid_record], "Duplicate task_id"),
            ([{**valid_record, "prompt_id": 2, "task_id": "2:1"}], "not present"),
            ([{**valid_record, "model_name": "deepseek-chat"}], "different model"),
        )
        for records, message in cases:
            with self.subTest(message=message):
                self.output_file.write_text(
                    "".join(json.dumps(record) + "\n" for record in records),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, message):
                    code_generation_api.load_completed_task_ids(
                        self.output_file,
                        expected_task_ids,
                        "qwen3-32b",
                    )

    def test_output_validation_rejects_invalid_middle_line(self):
        self.output_file.write_bytes(
            json.dumps(
                {
                    "prompt_id": 1,
                    "sample_index": 1,
                    "task_id": "1:1",
                    "model_name": "qwen3-32b",
                }
            ).encode("utf-8")
            + b"\ninvalid\n{}\n"
        )
        with self.assertRaisesRegex(ValueError, "Invalid output JSONL"):
            code_generation_api.load_completed_task_ids(
                self.output_file,
                {"1:1"},
                "qwen3-32b",
            )


if __name__ == "__main__":
    unittest.main()
