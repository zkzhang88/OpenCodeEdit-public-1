import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation.create_prompt_batch_infer import (
    BatchJsonlWriter,
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

    @staticmethod
    def part_path(output_file, part_number):
        return output_file.with_name(
            f"{output_file.stem}_part{part_number:03d}{output_file.suffix}"
        )

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

    @mock.patch("generation.create_prompt_batch_infer.MAX_BATCH_REQUESTS_PER_FILE", 2)
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
            sample_num=3,
            random_seed=42,
        )

        part_one = self.read_jsonl(self.part_path(output_file, 1))
        part_two = self.read_jsonl(self.part_path(output_file, 2))
        records = part_one + part_two
        self.assertEqual([len(part_one), len(part_two)], [2, 1])
        self.assertEqual(
            [record["custom_id"] for record in records],
            ["request-1", "request-2", "request-3"],
        )
        for request_number, record in enumerate(records, start=1):
            user_content = record["body"]["messages"][1]["content"]
            self.assert_batch_request(record, request_number, user_content)
            self.assertNotIn("second round", user_content)
            self.assertNotIn("prompt_id", record)
            self.assertNotIn("commit", record)

    @mock.patch("generation.create_prompt_batch_infer.MAX_BATCH_REQUESTS_PER_FILE", 2)
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

        records = self.read_jsonl(self.part_path(output_file, 1))
        self.assertFalse(self.part_path(output_file, 2).exists())
        self.assertEqual([record["custom_id"] for record in records], ["request-1", "request-2"])
        for request_number, record in enumerate(records, start=1):
            user_content = record["body"]["messages"][1]["content"]
            self.assert_batch_request(record, request_number, user_content)
            self.assertNotIn("prompt_id", record)
            self.assertNotIn("old_code", record)
            self.assertNotIn("new_code", record)

    @mock.patch("generation.create_prompt_batch_infer.MAX_BATCH_REQUESTS_PER_FILE", 2)
    def test_writer_creates_empty_part_and_cleans_previous_outputs(self):
        output_file = self.root / "prompts.jsonl"
        old_part_one = self.part_path(output_file, 1)
        old_part_four = self.part_path(output_file, 4)
        unrelated_file = self.root / "prompts_partABC.jsonl"
        output_file.write_text("legacy\n", encoding="utf-8")
        old_part_one.write_text("old one\n", encoding="utf-8")
        old_part_four.write_text("old four\n", encoding="utf-8")
        unrelated_file.write_text("keep\n", encoding="utf-8")

        with BatchJsonlWriter(output_file):
            pass

        self.assertFalse(output_file.exists())
        self.assertEqual(old_part_one.read_text(encoding="utf-8"), "")
        self.assertFalse(old_part_four.exists())
        self.assertTrue(unrelated_file.exists())

    def test_writer_splits_6001_requests_into_6000_and_one(self):
        output_file = self.root / "production_limit.jsonl"

        with BatchJsonlWriter(output_file) as output_writer:
            for request_number in range(1, 6002):
                output_writer.write(
                    {"custom_id": f"request-{request_number}"}
                )

        part_one = self.read_jsonl(self.part_path(output_file, 1))
        part_two = self.read_jsonl(self.part_path(output_file, 2))
        self.assertEqual([len(part_one), len(part_two)], [6000, 1])
        self.assertEqual(part_one[-1]["custom_id"], "request-6000")
        self.assertEqual(part_two[0]["custom_id"], "request-6001")
        self.assertFalse(self.part_path(output_file, 3).exists())


if __name__ == "__main__":
    unittest.main()
