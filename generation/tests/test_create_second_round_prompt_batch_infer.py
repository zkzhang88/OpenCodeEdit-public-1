import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from generation import create_second_round_prompt_batch_infer as converter


class CreateSecondRoundPromptBatchInferTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.first_round_base = self.root / "prompts.jsonl"
        self.results_dir = self.root / "results"
        self.output_dir = self.root / "round2"
        self.results_dir.mkdir()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def first_round_part(self, suffix):
        return self.root / f"prompts_part{suffix}.jsonl"

    @staticmethod
    def write_jsonl(path, records):
        path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )

    @staticmethod
    def read_jsonl(path):
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]

    @staticmethod
    def make_request(request_number, user_content=None):
        return {
            "custom_id": f"request-{request_number}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": user_content or f"first user {request_number}",
                    },
                ],
                "stream": False,
                "max_tokens": 2048,
                "temperature": 0.25,
            },
        }

    @staticmethod
    def make_result(request_number, content=None):
        return {
            "custom_id": f"request-{request_number}",
            "response": {
                "body": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": content
                                or f"assistant {request_number}",
                            }
                        }
                    ]
                }
            },
            "error": None,
        }

    def run_conversion(self):
        with mock.patch.object(
            converter,
            "get_prompts",
            return_value=("system", ["round one", "round two"]),
        ):
            return converter.create_second_round_batches(
                first_round_base_file=self.first_round_base,
                results_dir=self.results_dir,
                output_dir=self.output_dir,
            )

    def test_writes_full_conversations_in_numeric_file_and_record_order(self):
        request_two = self.make_request(2)
        self.write_jsonl(
            self.first_round_part("001"),
            [self.make_request(10), request_two],
        )
        self.write_jsonl(
            self.first_round_part("001_1"),
            [request_two],
        )
        self.write_jsonl(
            self.first_round_part("002"),
            [self.make_request(20)],
        )

        self.write_jsonl(
            self.results_dir / "batch_first.jsonl",
            [self.make_result(10), self.make_result(2)],
        )
        self.write_jsonl(
            self.results_dir / "batch_second.jsonl",
            [self.make_result(20)],
        )
        stale_output = (
            self.output_dir
            / "prompt_for_syn_batch_infer_round2_part003.jsonl"
        )
        unrelated_output = self.output_dir / "keep.jsonl"
        self.output_dir.mkdir()
        stale_output.write_text("stale\n", encoding="utf-8")
        unrelated_output.write_text("keep\n", encoding="utf-8")

        summaries = self.run_conversion()

        part_one_path = (
            self.output_dir
            / "prompt_for_syn_batch_infer_round2_part001.jsonl"
        )
        part_two_path = (
            self.output_dir
            / "prompt_for_syn_batch_infer_round2_part002.jsonl"
        )
        self.assertEqual(
            [summary["path"] for summary in summaries],
            [part_one_path, part_two_path],
        )
        self.assertEqual(
            [summary["count"] for summary in summaries],
            [2, 1],
        )
        part_one = self.read_jsonl(part_one_path)
        part_two = self.read_jsonl(part_two_path)
        self.assertEqual(
            [record["custom_id"] for record in part_one],
            ["request-2", "request-10"],
        )
        self.assertEqual(part_two[0]["custom_id"], "request-20")

        record = part_one[0]
        self.assertEqual(record["method"], "POST")
        self.assertEqual(record["url"], "/v1/chat/completions")
        self.assertEqual(record["body"]["model"], "test-model")
        self.assertEqual(record["body"]["temperature"], 0.25)
        self.assertEqual(
            [message["role"] for message in record["body"]["messages"]],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(
            [message["content"] for message in record["body"]["messages"]],
            ["system", "first user 2", "assistant 2", "round two"],
        )
        self.assertFalse(stale_output.exists())
        self.assertEqual(unrelated_output.read_text(encoding="utf-8"), "keep\n")

    def test_orders_result_files_by_their_minimum_numeric_custom_id(self):
        self.write_jsonl(
            self.first_round_part("001"),
            [self.make_request(2), self.make_request(100)],
        )
        self.write_jsonl(
            self.results_dir / "a_later_name.jsonl",
            [self.make_result(100)],
        )
        self.write_jsonl(
            self.results_dir / "z_earlier_name.jsonl",
            [self.make_result(2)],
        )

        self.run_conversion()

        part_one = self.read_jsonl(
            self.output_dir
            / "prompt_for_syn_batch_infer_round2_part001.jsonl"
        )
        part_two = self.read_jsonl(
            self.output_dir
            / "prompt_for_syn_batch_infer_round2_part002.jsonl"
        )
        self.assertEqual(part_one[0]["custom_id"], "request-2")
        self.assertEqual(part_two[0]["custom_id"], "request-100")

    def test_accepts_a_single_result_file(self):
        self.write_jsonl(
            self.first_round_part("001"),
            [self.make_request(1)],
        )
        result_file = self.root / "batch.jsonl"
        self.write_jsonl(result_file, [self.make_result(1)])

        with mock.patch.object(
            converter,
            "get_prompts",
            return_value=("system", ["round one", "round two"]),
        ):
            summaries = converter.create_second_round_batches(
                first_round_base_file=self.first_round_base,
                results_dir=result_file,
                output_dir=self.output_dir,
            )

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["count"], 1)
        self.assertEqual(
            self.read_jsonl(summaries[0]["path"])[0]["custom_id"],
            "request-1",
        )

    def test_rejects_conflicting_duplicate_first_round_request(self):
        self.write_jsonl(
            self.first_round_part("001"),
            [self.make_request(1)],
        )
        self.write_jsonl(
            self.first_round_part("001_1"),
            [self.make_request(1, user_content="different")],
        )
        self.write_jsonl(
            self.results_dir / "batch.jsonl",
            [self.make_result(1)],
        )

        with self.assertRaisesRegex(
            converter.BatchConversionError,
            "conflicting first-round request",
        ):
            self.run_conversion()

    def test_rejects_duplicate_result(self):
        self.write_jsonl(
            self.first_round_part("001"),
            [self.make_request(1)],
        )
        self.write_jsonl(
            self.results_dir / "batch_one.jsonl",
            [self.make_result(1)],
        )
        self.write_jsonl(
            self.results_dir / "batch_two.jsonl",
            [self.make_result(1)],
        )

        with self.assertRaisesRegex(
            converter.BatchConversionError,
            "duplicate result",
        ):
            self.run_conversion()

    def test_rejects_missing_first_round_request_without_touching_output(self):
        self.write_jsonl(
            self.first_round_part("001"),
            [self.make_request(1)],
        )
        self.write_jsonl(
            self.results_dir / "batch.jsonl",
            [self.make_result(2)],
        )
        existing_output = (
            self.output_dir
            / "prompt_for_syn_batch_infer_round2_part001.jsonl"
        )
        self.output_dir.mkdir()
        existing_output.write_text("existing\n", encoding="utf-8")

        with self.assertRaisesRegex(
            converter.BatchConversionError,
            "Missing first-round requests for: request-2",
        ):
            self.run_conversion()

        self.assertEqual(
            existing_output.read_text(encoding="utf-8"),
            "existing\n",
        )

    def test_rejects_interleaved_result_file_ranges(self):
        self.write_jsonl(
            self.first_round_part("001"),
            [
                self.make_request(1),
                self.make_request(5),
                self.make_request(10),
            ],
        )
        self.write_jsonl(
            self.results_dir / "outer.jsonl",
            [self.make_result(1), self.make_result(10)],
        )
        self.write_jsonl(
            self.results_dir / "inner.jsonl",
            [self.make_result(5)],
        )

        with self.assertRaisesRegex(
            converter.BatchConversionError,
            "overlap or interleave",
        ):
            self.run_conversion()

    def test_rejects_invalid_result_shapes(self):
        cases = {
            "invalid custom ID": (
                {
                    **self.make_result(1),
                    "custom_id": "request-one",
                },
                "invalid custom_id",
            ),
            "API error": (
                {
                    **self.make_result(1),
                    "error": {"message": "failed"},
                },
                "contains an error",
            ),
            "empty choices": (
                {
                    **self.make_result(1),
                    "response": {"body": {"choices": []}},
                },
                "exactly one choice",
            ),
            "wrong role": (
                {
                    **self.make_result(1),
                    "response": {
                        "body": {
                            "choices": [
                                {
                                    "message": {
                                        "role": "user",
                                        "content": "content",
                                    }
                                }
                            ]
                        }
                    },
                },
                "must be 'assistant'",
            ),
            "empty content": (
                self.make_result(1, content=" "),
                "must be a non-empty string",
            ),
        }

        for label, (record, expected_error) in cases.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    converter.BatchConversionError,
                    expected_error,
                ):
                    converter.validate_result_record(record, "result.jsonl:1")

    def test_rejects_result_file_over_batch_limit(self):
        self.write_jsonl(
            self.results_dir / "batch.jsonl",
            [self.make_result(1), self.make_result(2), self.make_result(3)],
        )

        with mock.patch.object(
            converter,
            "MAX_BATCH_REQUESTS_PER_FILE",
            2,
        ):
            with self.assertRaisesRegex(
                converter.BatchConversionError,
                "maximum batch size is 2",
            ):
                converter.load_result_batches(self.results_dir)


if __name__ == "__main__":
    unittest.main()
