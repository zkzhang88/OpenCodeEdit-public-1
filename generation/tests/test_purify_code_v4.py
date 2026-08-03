import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.purify_code_v4 import purify_code_from_jsonl


class PurifyCodeTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_file = self.root / "input.jsonl"
        self.output_file = self.root / "output.jsonl"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_records(self, records):
        self.input_file.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )

    def read_output(self):
        return [
            json.loads(line)
            for line in self.output_file.read_text(encoding="utf-8").splitlines()
        ]

    def test_writes_record_only_when_all_fields_are_purified(self):
        self.write_records(
            [
                {
                    "id": "valid",
                    "before": "```python\nvalue = 1\n```",
                    "after": (
                        "```markdown\nexplanation\n```\n"
                        "```python\nvalue = 2\n```"
                    ),
                },
                {
                    "id": "partial",
                    "before": "```python\nvalue = 1\n```",
                    "after": "```python\nvalue = 2",
                },
            ]
        )
        messages = io.StringIO()

        with redirect_stdout(messages):
            purify_code_from_jsonl(
                self.input_file,
                self.output_file,
                purify_fields=["before", "after"],
            )

        output = self.read_output()
        self.assertEqual([record["id"] for record in output], ["valid"])
        self.assertEqual(output[0]["before_purify"], "value = 1")
        self.assertEqual(output[0]["after_purify"], "value = 2")
        self.assertNotIn("partial", self.output_file.read_text(encoding="utf-8"))
        self.assertIn(f"{self.input_file}:2", messages.getvalue())
        self.assertIn("field 'after': No complete code block found", messages.getvalue())
        self.assertIn("total=2, written=1, dropped=1", messages.getvalue())

    def test_drops_missing_non_string_and_empty_code_fields(self):
        self.write_records(
            [
                {"id": "missing"},
                {"id": "non-string", "code": None},
                {"id": "empty", "code": "```python\n   \n```"},
            ]
        )
        messages = io.StringIO()

        with redirect_stdout(messages):
            purify_code_from_jsonl(
                self.input_file,
                self.output_file,
                purify_field="code",
            )

        self.assertEqual(self.read_output(), [])
        output = messages.getvalue()
        self.assertIn(f"{self.input_file}:1", output)
        self.assertIn("Field is missing", output)
        self.assertIn(f"{self.input_file}:2", output)
        self.assertIn("Expected a string, got NoneType", output)
        self.assertIn(f"{self.input_file}:3", output)
        self.assertIn("Extracted code block is empty", output)
        self.assertIn("total=3, written=0, dropped=3", output)

    def test_keep_language_mark_behavior_is_preserved(self):
        self.write_records([{"code_after": "```python\nvalue = 1\n```"}])

        purify_code_from_jsonl(
            self.input_file,
            self.output_file,
            keep_language_mark=True,
        )

        self.assertEqual(
            self.read_output()[0]["code_after_purify"],
            "## python\nvalue = 1",
        )


if __name__ == "__main__":
    unittest.main()
