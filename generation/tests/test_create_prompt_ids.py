import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation.create_prompt import create_prompt, create_prompt_rewrite_commit


class CreatePromptIdTests(unittest.TestCase):
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
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    @mock.patch("generation.create_prompt.get_prompts")
    def test_code_extension_ids_follow_output_order(self, get_prompts):
        get_prompts.return_value = (
            "system",
            [
                "{code_snippet_1}\n{code_snippet_2}\n{code_before_shot}\n"
                "{desc_instr_shot}\n{lazy_instr_shot}",
                "second round",
            ],
        )
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

        self.assertEqual(
            [record["prompt_id"] for record in self.read_jsonl(output_file)],
            [1, 2, 3],
        )

    @mock.patch("generation.create_prompt.get_prompts")
    def test_rewrite_ids_are_assigned_after_filtering_and_shuffle(self, get_prompts):
        get_prompts.return_value = (
            "system",
            [
                "{code_before}\n{code_diff}\n{commit_message}\n"
                "{desc_instr_shot}\n{lazy_instr_shot}"
            ],
        )
        commit_file = self.root / "commits.jsonl"
        records = [
            {
                "commit": "one",
                "old_contents": "old one",
                "new_contents": "new one",
                "message": "message one",
            },
            {
                "commit": "skipped",
                "old_contents": "old",
                "new_contents": "",
                "message": "message",
            },
            {
                "commit": "two",
                "old_contents": "old two",
                "new_contents": "new two",
                "message": "message two",
            },
        ]
        commit_file.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        output_file = self.root / "rewrite.jsonl"

        create_prompt_rewrite_commit(
            commit_file,
            self.oneshot_file,
            "v5.9",
            output_file,
            shuffle=True,
        )

        output_records = self.read_jsonl(output_file)
        self.assertEqual([record["prompt_id"] for record in output_records], [1, 2])
        self.assertEqual({record["commit"] for record in output_records}, {"one", "two"})


if __name__ == "__main__":
    unittest.main()
