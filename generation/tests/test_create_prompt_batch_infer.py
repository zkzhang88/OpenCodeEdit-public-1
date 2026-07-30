import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation.create_prompt_batch_infer import (
    create_prompt,
    create_prompt_rewrite_commit,
)


class CreatePromptBatchInferTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.oneshot_file = self.root / "oneshot.jsonl"
        self.oneshot_file.write_text(
            json.dumps(
                {
                    "code_before": "before",
                    "instruct_descriptive": "describe",
                    "instruct_lazy": "lazy",
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def read_jsonl(path):
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]

    def assert_batch_request(self, record, request_number, user_content):
        self.assertEqual(
            set(record),
            {"custom_id", "method", "url", "body"},
        )
        self.assertEqual(record["custom_id"], f"request-{request_number}")
        self.assertEqual(record["method"], "POST")
        self.assertEqual(record["url"], "/v1/chat/completions")
        self.assertEqual(
            record["body"],
            {
                "model": "deepseek-ai/DeepSeek-V3",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": user_content},
                ],
                "stream": False,
                "max_tokens": 2048,
            },
        )

    @mock.patch("generation.create_prompt_batch_infer.get_prompts")
    def test_code_extension_writes_first_round_batch_requests(self, get_prompts):
        first_round = (
            "{code_snippet_1}\n{code_snippet_2}\n{code_before_shot}\n"
            "{desc_instr_shot}\n{lazy_instr_shot}"
        )
        get_prompts.return_value = ("system", [first_round, "second round"])
        commit_file = self.root / "commits.jsonl"
        commit_file.write_text(
            "".join(
                json.dumps(
                    {
                        "commit": f"commit-{index}",
                        "old_contents": "\n".join(
                            f"line-{line}" for line in range(6)
                        ),
                        "new_contents": "new code",
                        "message": "message",
                    }
                )
                + "\n"
                for index in range(2)
            ),
            encoding="utf-8",
        )
        output_file = self.root / "prompts.jsonl"

        create_prompt(
            commit_file,
            self.oneshot_file,
            "v5.2",
            output_file,
            min_snippet_lines=2,
            max_snippet_lines=3,
            sample_num=2,
            random_seed=42,
        )

        records = self.read_jsonl(output_file)
        self.assertEqual([record["custom_id"] for record in records], ["request-1", "request-2"])
        for request_number, record in enumerate(records, start=1):
            user_content = record["body"]["messages"][1]["content"]
            self.assert_batch_request(record, request_number, user_content)
            self.assertNotIn("second round", user_content)
            self.assertNotIn("prompt_id", record)
            self.assertNotIn("commit", record)

    @mock.patch("generation.create_prompt_batch_infer.get_prompts")
    def test_rewrite_writes_single_round_batch_requests(self, get_prompts):
        user_template = (
            "{code_before}\n{code_diff}\n{commit_message}\n"
            "{desc_instr_shot}\n{lazy_instr_shot}"
        )
        get_prompts.return_value = ("system", [user_template])
        commit_file = self.root / "commits.jsonl"
        commit_file.write_text(
            "".join(
                json.dumps(
                    {
                        "commit": str(index),
                        "old_contents": f"old {index}",
                        "new_contents": f"new {index}",
                        "message": f"message {index}",
                    }
                )
                + "\n"
                for index in range(2)
            ),
            encoding="utf-8",
        )
        output_file = self.root / "rewrite.jsonl"

        create_prompt_rewrite_commit(
            commit_file,
            self.oneshot_file,
            "v5.9",
            output_file,
            shuffle=True,
            random_seed=42,
        )

        records = self.read_jsonl(output_file)
        self.assertEqual([record["custom_id"] for record in records], ["request-1", "request-2"])
        for request_number, record in enumerate(records, start=1):
            user_content = record["body"]["messages"][1]["content"]
            self.assert_batch_request(record, request_number, user_content)
            self.assertNotIn("prompt_id", record)
            self.assertNotIn("old_code", record)
            self.assertNotIn("new_code", record)


if __name__ == "__main__":
    unittest.main()
