from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from generation.prompts_for_check import get_prompts
from generation.semantic_check import _load_and_prepare_input
from generation.split_instructions import main, split_instructions


def write_jsonl(path: Path, records: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class SplitInstructionsTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_file = self.root / "triplets_quality_filtered.jsonl"
        self.records = [
            {
                "commit": ["before-1", "after-1"],
                "code_snippet": ["old", "new"],
                "code_before_purify": "mapping = {'before': 1}\n",
                "code_after_purify": "mapping = {'after': 2}\n",
                "instruct_descriptive_purify": (
                    "详细更新映射。\nKeep the existing API."
                ),
                "instruct_lazy_purify": "更新映射。",
                "instruct_purify": "stale instruction",
                "instr_type": "stale_type",
                "custom_metadata": {"source": "测试"},
            },
            {
                "commit": "commit-2",
                "code_before_purify": "value = 1\n",
                "code_after_purify": "value = 2\n",
                "instruct_descriptive_purify": "Increment the value to two.",
                "instruct_lazy_purify": "Increment it.",
            },
        ]
        write_jsonl(self.input_file, self.records)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_splits_all_records_preserves_fields_and_overrides_normalized_values(self):
        summary = split_instructions(
            input_file=self.input_file,
            model_name="ds",
        )
        descriptive_file = Path(summary["descriptive_file"])
        lazy_file = Path(summary["lazy_file"])
        self.assertEqual(
            descriptive_file,
            self.root / "triplets_quality_filtered_descriptive.jsonl",
        )
        self.assertEqual(
            lazy_file,
            self.root / "triplets_quality_filtered_lazy.jsonl",
        )

        descriptive = read_jsonl(descriptive_file)
        lazy = read_jsonl(lazy_file)
        self.assertEqual(len(descriptive), len(self.records))
        self.assertEqual(len(lazy), len(self.records))
        self.assertEqual(
            [record["commit"] for record in descriptive],
            [record["commit"] for record in self.records],
        )
        self.assertEqual(
            [record["commit"] for record in lazy],
            [record["commit"] for record in self.records],
        )
        self.assertEqual(
            descriptive[0]["instruct_purify"],
            self.records[0]["instruct_descriptive_purify"],
        )
        self.assertEqual(
            lazy[0]["instruct_purify"],
            self.records[0]["instruct_lazy_purify"],
        )
        self.assertTrue(
            all(record["instr_type"] == "ds_descriptive" for record in descriptive)
        )
        self.assertTrue(all(record["instr_type"] == "ds_lazy" for record in lazy))
        for source, descriptive_record, lazy_record in zip(
            self.records, descriptive, lazy
        ):
            for key, value in source.items():
                if key not in {"instruct_purify", "instr_type"}:
                    self.assertEqual(descriptive_record[key], value)
                    self.assertEqual(lazy_record[key], value)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["descriptive_instr_type"], "ds_descriptive")
        self.assertEqual(summary["lazy_instr_type"], "ds_lazy")
        self.assertEqual(len(summary["input_sha256"]), 64)

    def test_both_outputs_match_semantic_check_default_input_contract(self):
        summary = split_instructions(
            input_file=self.input_file,
            model_name="qwen3",
        )
        system_prompt, user_prompt = get_prompts()
        for key, expected_type in (
            ("descriptive_file", "qwen3_descriptive"),
            ("lazy_file", "qwen3_lazy"),
        ):
            records, prompts = _load_and_prepare_input(
                Path(summary[key]),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                pre_field="code_before_purify",
                post_field="code_after_purify",
                instruction_field="instruct_purify",
            )
            self.assertEqual(len(records), 2)
            self.assertEqual(len(prompts), 2)
            self.assertTrue(all(record["instr_type"] == expected_type for record in records))
            self.assertTrue(all(prompt["instr_type"] == expected_type for prompt in prompts))

    def test_custom_fields_outputs_and_cli(self):
        custom_input = self.root / "custom.jsonl"
        write_jsonl(
            custom_input,
            [
                {
                    "code_before_purify": "x = 1",
                    "code_after_purify": "x = 2",
                    "long_instruction": "Set x to two with explanation.",
                    "short_instruction": "Set x to two.",
                }
            ],
        )
        descriptive_file = self.root / "outputs" / "long.jsonl"
        lazy_file = self.root / "outputs" / "short.jsonl"
        self.assertEqual(
            main(
                [
                    "--input-file",
                    str(custom_input),
                    "--model-name",
                    "qwen3",
                    "--descriptive-file",
                    str(descriptive_file),
                    "--lazy-file",
                    str(lazy_file),
                    "--descriptive-field",
                    "long_instruction",
                    "--lazy-field",
                    "short_instruction",
                ]
            ),
            0,
        )
        self.assertEqual(
            read_jsonl(descriptive_file)[0]["instr_type"],
            "qwen3_descriptive",
        )
        self.assertEqual(read_jsonl(lazy_file)[0]["instr_type"], "qwen3_lazy")

    def test_rejects_invalid_input_without_final_outputs(self):
        cases = {
            "missing": json.dumps(
                {
                    "instruct_lazy_purify": "lazy",
                    "code_before_purify": "x = 1",
                    "code_after_purify": "x = 2",
                }
            )
            + "\n",
            "empty": json.dumps(
                {
                    "instruct_descriptive_purify": " ",
                    "instruct_lazy_purify": "lazy",
                }
            )
            + "\n",
            "non_string": json.dumps(
                {
                    "instruct_descriptive_purify": ["invalid"],
                    "instruct_lazy_purify": "lazy",
                }
            )
            + "\n",
            "blank_line": json.dumps(self.records[0], ensure_ascii=False) + "\n\n",
            "invalid_json": "{not json}\n",
            "non_object": "[]\n",
            "empty_file": "",
        }
        for name, content in cases.items():
            with self.subTest(name=name):
                case_root = self.root / name
                case_root.mkdir()
                input_file = case_root / "input.jsonl"
                input_file.write_text(content, encoding="utf-8")
                with self.assertRaises((OSError, ValueError)):
                    split_instructions(input_file=input_file, model_name="ds")
                self.assertFalse((case_root / "input_descriptive.jsonl").exists())
                self.assertFalse((case_root / "input_lazy.jsonl").exists())
                self.assertEqual(list(case_root.glob(".*.tmp")), [])

    def test_rejects_invalid_model_names_path_conflicts_and_existing_outputs(self):
        for model_name in (
            "",
            "../ds",
            "deepseek/v3",
            "deepseek-v3",
            " ds",
            "模型",
        ):
            with self.subTest(model_name=model_name):
                with self.assertRaisesRegex(ValueError, "model_name"):
                    split_instructions(
                        input_file=self.input_file,
                        model_name=model_name,
                    )

        with self.assertRaisesRegex(ValueError, "must be distinct"):
            split_instructions(
                input_file=self.input_file,
                model_name="ds",
                descriptive_file=self.input_file,
            )

        descriptive_file = self.root / "existing.jsonl"
        descriptive_file.write_text("do not overwrite\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "already exists"):
            split_instructions(
                input_file=self.input_file,
                model_name="ds",
                descriptive_file=descriptive_file,
                lazy_file=self.root / "new_lazy.jsonl",
            )
        self.assertEqual(
            descriptive_file.read_text(encoding="utf-8"), "do not overwrite\n"
        )
        self.assertFalse((self.root / "new_lazy.jsonl").exists())

    def test_cleans_temporary_files_when_writing_fails(self):
        with mock.patch(
            "generation.split_instructions._write_split_records",
            side_effect=RuntimeError("write failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                split_instructions(input_file=self.input_file, model_name="ds")
        self.assertFalse(
            (self.root / "triplets_quality_filtered_descriptive.jsonl").exists()
        )
        self.assertFalse((self.root / "triplets_quality_filtered_lazy.jsonl").exists())
        self.assertEqual(list(self.root.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
