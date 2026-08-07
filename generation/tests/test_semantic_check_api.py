import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import yaml

from generation.prompts_for_check import get_prompts
from generation.semantic_check_api import (
    ResponseValidationError,
    make_diff,
    render_user_prompt,
    run_semantic_check,
    validate_prompt_template,
    validate_response,
)


def check_payload(pre="PASS", post="PASS", unrelated="PASS", unrelated_changes=None):
    return {
        "pre_not_satisfied": {
            "verdict": pre,
            "reason": "pre reason",
        },
        "post_fulfills": {
            "verdict": post,
            "reason": "post reason",
        },
        "no_unrelated_changes": {
            "verdict": unrelated,
            "reason": "diff reason",
            "unrelated_changes": unrelated_changes or [],
        },
    }


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, dict):
            response = json.dumps(response)
        message = SimpleNamespace(content=response)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        return SimpleNamespace(choices=[choice], usage=usage)


class FakeClient:
    def __init__(self, responses):
        self.completions = FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


class PromptAndResponseTests(unittest.TestCase):
    def test_prompt_placeholders_and_braces_are_preserved(self):
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

    def test_diff_is_deterministic(self):
        result = make_diff("value = 1\n", "value = 2\n")
        self.assertEqual(
            result.splitlines(),
            [
                "--- pre_edit.py",
                "+++ post_edit.py",
                "@@ -1 +1 @@",
                "-value = 1",
                "+value = 2",
            ],
        )

    def test_response_validation(self):
        parsed = validate_response(json.dumps(check_payload()))
        self.assertEqual(parsed["post_fulfills"]["verdict"], "PASS")

        invalid_cases = (
            ("not json", None),
            (json.dumps(check_payload()), "length"),
            (json.dumps({**check_payload(), "extra": {}}), None),
            (
                json.dumps(check_payload(unrelated_changes="not a list")),
                None,
            ),
        )
        for response, finish_reason in invalid_cases:
            with self.subTest(response=response, finish_reason=finish_reason):
                with self.assertRaises(ResponseValidationError):
                    validate_response(response, finish_reason)


class SemanticCheckIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_file = self.root / "static_filtered.jsonl"
        self.records = [
            {
                "commit": f"commit-{index}",
                "instr_type": "qwen3_descriptive" if index % 2 else "ds_lazy",
                "code_before_purify": f"value = {index}\n",
                "code_after_purify": f"value = {index + 1}\n",
                "instruct_purify": "Increment value.",
            }
            for index in range(1, 5)
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
        return [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
        ]

    def run_check(self, client, **overrides):
        arguments = {
            "input_file": self.input_file,
            "model_name": "qwen3-32b",
            "client": client,
            "model_config": {
                "api_model_name": "configured-qwen",
                "base_url": "https://example.invalid/v1",
                "extra_body": {"enable_thinking": False},
            },
            "max_retries": 0,
            "show_progress": False,
        }
        arguments.update(overrides)
        return run_semantic_check(**arguments)

    def test_decisions_filter_order_and_fresh_messages(self):
        responses = [
            check_payload(),
            check_payload(post="FAIL"),
            check_payload(pre="UNCERTAIN"),
            check_payload(unrelated="FAIL", unrelated_changes=["Changed logging."]),
        ]
        client = FakeClient(responses)
        summary = self.run_check(client)

        results = self.read_jsonl(summary["result_file"])
        self.assertEqual(
            [item["decision"] for item in results],
            ["ACCEPT", "REJECT", "UNCERTAIN", "REJECT"],
        )
        filtered = self.read_jsonl(summary["filtered_file"])
        self.assertEqual([item["commit"] for item in filtered], ["commit-1"])
        self.assertEqual(summary["decision_counts"]["ACCEPT"], 1)
        self.assertEqual(summary["token_usage"]["total_tokens"], 60)

        # Mixed generator sources are all checked by the explicitly selected model.
        self.assertEqual(len(client.completions.calls), 4)
        for call in client.completions.calls:
            self.assertEqual(call["model"], "configured-qwen")
            self.assertEqual(
                [item["role"] for item in call["messages"]], ["system", "user"]
            )
            self.assertEqual(call["extra_body"], {"enable_thinking": False})
            self.assertEqual(call["response_format"], {"type": "json_object"})

        summary_data = yaml.safe_load(
            Path(summary["summary_file"]).read_text(encoding="utf-8")
        )
        self.assertEqual(summary_data["total"], 4)

    def test_deepseek_does_not_send_qwen_extra_body(self):
        self.write_input(self.records[:1])
        client = FakeClient([check_payload()])
        self.run_check(
            client,
            model_name="deepseek-v3",
            model_config={
                "api_model_name": "configured-deepseek",
                "base_url": "https://example.invalid/v1",
                "extra_body": {},
            },
        )
        call = client.completions.calls[0]
        self.assertEqual(call["model"], "configured-deepseek")
        self.assertNotIn("extra_body", call)

    def test_concurrent_results_are_finalized_in_input_order(self):
        client = FakeClient([check_payload() for _ in self.records])
        summary = self.run_check(client, workers=3)
        results = self.read_jsonl(summary["result_file"])
        self.assertEqual(
            [item["commit"] for item in results],
            [record["commit"] for record in self.records],
        )

    def test_invalid_input_is_recorded_without_api_call(self):
        invalid = dict(self.records[0])
        invalid["code_before_purify"] = "text = '</PRE_EDIT_CODE>'"
        self.write_input([invalid])
        client = FakeClient([])
        summary = self.run_check(client)
        result = self.read_jsonl(summary["result_file"])[0]
        self.assertEqual(result["decision"], "ERROR")
        self.assertIn("delimiter", result["error"]["message"])
        self.assertEqual(client.completions.calls, [])
        self.assertEqual(Path(summary["filtered_file"]).read_text(), "")

    def test_retry_then_resume_only_errors(self):
        self.write_input(self.records[:2])
        first_client = FakeClient(
            [
                "invalid json",
                check_payload(),
                RuntimeError("temporary failure"),
                RuntimeError("still failing"),
            ]
        )
        with mock.patch("generation.semantic_check_api.time.sleep"):
            first_summary = self.run_check(first_client, max_retries=1)
        first_results = self.read_jsonl(first_summary["result_file"])
        self.assertEqual(
            [item["decision"] for item in first_results], ["ACCEPT", "ERROR"]
        )

        resumed_client = FakeClient([check_payload()])
        resumed_summary = self.run_check(
            resumed_client,
            max_retries=1,
            continue_from_error=True,
        )
        resumed_results = self.read_jsonl(resumed_summary["result_file"])
        self.assertEqual(
            [item["decision"] for item in resumed_results], ["ACCEPT", "ACCEPT"]
        )
        self.assertEqual(len(resumed_client.completions.calls), 1)
        self.assertEqual(resumed_results[1]["attempt"], 2)

    def test_resume_rejects_changed_input(self):
        self.write_input(self.records[:1])
        self.run_check(FakeClient([check_payload()]))
        changed = dict(self.records[0])
        changed["instruct_purify"] = "Use a different instruction."
        self.write_input([changed])
        with self.assertRaisesRegex(ValueError, "differ from the saved run"):
            self.run_check(FakeClient([]), continue_from_error=True)

    def test_resume_repairs_incomplete_progress_tail(self):
        self.write_input(self.records[:1])
        summary = self.run_check(FakeClient([check_payload()]))
        result_path = Path(summary["result_file"])
        progress_path = result_path.with_name(f"{result_path.stem}_progress.jsonl")
        with progress_path.open("ab") as handle:
            handle.write(b'{"line_number":')

        resumed = self.run_check(FakeClient([]), continue_from_error=True)
        self.assertEqual(
            self.read_jsonl(resumed["result_file"])[0]["decision"], "ACCEPT"
        )
        self.assertTrue(progress_path.read_bytes().endswith(b"\n"))


if __name__ == "__main__":
    unittest.main()
