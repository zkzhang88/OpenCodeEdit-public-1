import json
from pathlib import Path
import tempfile
import unittest

from generation.export_quality_issue_code import (
    export_issue_code,
    infer_input_file,
    main,
)


class ExportQualityIssueCodeTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_file = self.root / "sample.jsonl"
        self.issues_file = self.root / "sample_quality_issues.jsonl"
        self.records = [
            {
                "code_before_purify": "value = 1\n",
                "code_after_purify": "value = 2\n",
                "instruct_purify": "Update the value.",
            },
            {
                "code_before_purify": "def broken():\n",
                "code_after_purify": "def fixed():\n    pass\n",
                "instruct_purify": "Complete the function.",
            },
            {
                "code_before_purify": "result = old_name\n",
                "code_after_purify": "result = missing_name\n",
                "instruct_purify": "Rename the result source.",
            },
        ]
        self.input_file.write_text(
            "".join(json.dumps(record) + "\n" for record in self.records),
            encoding="utf-8",
        )
        issue_records = [
            {"line_number": 2, "issues": [{"code": "incomplete_structure"}]},
            {"line_number": 3, "issues": [{"code": "new_undefined_name"}]},
        ]
        self.issues_file.write_text(
            "".join(json.dumps(record) + "\n" for record in issue_records),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_infers_source_file_from_quality_report_name(self):
        self.assertEqual(infer_input_file(self.issues_file), self.input_file)

    def test_exports_selected_lines_as_separate_code_files(self):
        output_dir = self.root / "exports"
        exported = export_issue_code(
            self.input_file, self.issues_file, [3, 2], output_dir
        )

        self.assertEqual(
            exported,
            [output_dir / "line_000002", output_dir / "line_000003"],
        )
        self.assertEqual(
            (output_dir / "line_000002" / "pre_edit.py").read_text(
                encoding="utf-8"
            ),
            self.records[1]["code_before_purify"],
        )
        self.assertEqual(
            (output_dir / "line_000003" / "post_edit.py").read_text(
                encoding="utf-8"
            ),
            self.records[2]["code_after_purify"],
        )
        metadata = json.loads(
            (output_dir / "line_000003" / "issues.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(metadata["line_number"], 3)
        self.assertEqual(
            metadata["edit_instruction"], self.records[2]["instruct_purify"]
        )

    def test_reports_missing_issue_line(self):
        with self.assertRaisesRegex(ValueError, "absent from the quality report"):
            export_issue_code(
                self.input_file, self.issues_file, [1], self.root / "exports"
            )

    def test_cli_uses_inferred_input_and_default_output_directory(self):
        self.assertEqual(
            main(
                [
                    "--issues-file",
                    str(self.issues_file),
                    "--line-number",
                    "2",
                ]
            ),
            0,
        )
        output = self.root / "sample_quality_issue_code" / "line_000002"
        self.assertTrue((output / "pre_edit.py").is_file())
        self.assertTrue((output / "post_edit.py").is_file())


if __name__ == "__main__":
    unittest.main()
