import json
from pathlib import Path
import tempfile
import unittest

from generation.export_filtered_samples import export_first_samples, main


class ExportFilteredSamplesTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_file = self.root / "filtered.jsonl"
        self.records = [
            {
                "code_before_purify": "value = 1\n",
                "code_after_purify": "value = 2\n",
                "instruct_purify": "Update the value.",
            },
            {
                "code_before_purify": "def old():\n    pass\n",
                "code_after_purify": "def new():\n    return 1\n",
                "instruct_purify": "Implement and rename the function.\n",
            },
            {
                "code_before_purify": "third = False\n",
                "code_after_purify": "third = True\n",
                "instruct_purify": "This record should not be exported.",
            },
        ]
        self.input_file.write_text(
            "".join(json.dumps(record) + "\n" for record in self.records),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_exports_only_first_n_records(self):
        output_dir = self.root / "samples"
        exported = export_first_samples(self.input_file, 2, output_dir)

        self.assertEqual(
            exported,
            [output_dir / "line_000001", output_dir / "line_000002"],
        )
        self.assertEqual(
            (exported[0] / "pre_edit.py").read_text(encoding="utf-8"),
            "value = 1\n",
        )
        self.assertEqual(
            (exported[1] / "post_edit.py").read_text(encoding="utf-8"),
            "def new():\n    return 1\n",
        )
        self.assertEqual(
            (exported[1] / "instruction.txt").read_text(encoding="utf-8"),
            "Implement and rename the function.\n",
        )
        self.assertFalse((output_dir / "line_000003").exists())

    def test_rejects_non_positive_count(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            export_first_samples(self.input_file, 0, self.root / "samples")

    def test_reports_when_input_has_fewer_than_n_records(self):
        with self.assertRaisesRegex(ValueError, "contains only 3 lines"):
            export_first_samples(self.input_file, 4, self.root / "samples")

    def test_cli_exports_records(self):
        output_dir = self.root / "cli-output"
        self.assertEqual(
            main(
                [
                    "1",
                    "--input-file",
                    str(self.input_file),
                    "--output-dir",
                    str(output_dir),
                ]
            ),
            0,
        )
        self.assertTrue((output_dir / "line_000001" / "pre_edit.py").is_file())


if __name__ == "__main__":
    unittest.main()
