import json
from pathlib import Path
import tempfile
import unittest

from generation.export_filtered_samples import (
    MANUAL_REVIEW_FIELDS,
    export_first_samples,
    main,
)


class ExportFilteredSamplesTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_file = self.root / "filtered.jsonl"
        self.records = [
            {
                "commit": "commit-1",
                "code_before_purify": "value = 1\n",
                "code_after_purify": "value = 2\n",
                "instruct_purify": "Update the value.",
                "instr_type": "descriptive",
            },
            {
                "commit": "commit-2",
                "code_before_purify": "def old():\n    pass\n",
                "code_after_purify": "def new():\n    return 1\n",
                "instruct_purify": "Implement and rename the function.\n",
                "instr_type": "lazy",
            },
            {
                "commit": "commit-3",
                "code_before_purify": "third = False\n",
                "code_after_purify": "third = True\n",
                "instruct_purify": "This record should not be exported.",
                "instr_type": "descriptive",
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
        old_record_dir = output_dir / "line_000002"
        old_record_dir.mkdir(parents=True)
        (old_record_dir / "instruction.txt").write_text(
            "stale instruction", encoding="utf-8"
        )
        (old_record_dir / "instruction.jsonl").write_text(
            '{"stale": true}\n', encoding="utf-8"
        )
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
        instruction_record = json.loads(
            (exported[1] / "instruction.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            instruction_record,
            {
                "instruct_purify": "Implement and rename the function.\n",
                "commit": "commit-2",
                "instr_type": "lazy",
                **{field: None for field in MANUAL_REVIEW_FIELDS},
            },
        )
        self.assertEqual(
            list(instruction_record)[-len(MANUAL_REVIEW_FIELDS) :],
            list(MANUAL_REVIEW_FIELDS),
        )
        self.assertEqual(
            [key for key in instruction_record if key.startswith("manual_")],
            ["manual_post_edit_fulfills_instruction"],
        )
        self.assertIsNone(
            instruction_record["manual_post_edit_fulfills_instruction"]
        )
        self.assertEqual(
            (exported[0] / "instruction.txt").read_text(encoding="utf-8"),
            "Update the value.\n",
        )
        self.assertEqual(
            (exported[1] / "instruction.txt").read_text(encoding="utf-8"),
            "Implement and rename the function.\n",
        )
        self.assertFalse((exported[1] / "instruction.jsonl").exists())
        instruction_text = (exported[1] / "instruction.json").read_text(
            encoding="utf-8"
        )
        self.assertTrue(instruction_text.startswith("{\n  \"instruct_purify\""))
        self.assertTrue(instruction_text.endswith("\n"))
        self.assertFalse((output_dir / "line_000003").exists())

    def test_rejects_non_positive_count(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            export_first_samples(self.input_file, 0, self.root / "samples")

    def test_exports_first_k_records_for_each_requested_type(self):
        output_dir = self.root / "by-type"
        exported = export_first_samples(
            self.input_file,
            1,
            output_dir,
            instr_types=["descriptive", "lazy"],
        )

        self.assertEqual(
            exported,
            [
                output_dir / "descriptive" / "line_000001",
                output_dir / "lazy" / "line_000002",
            ],
        )
        for record_dir in exported:
            self.assertTrue((record_dir / "pre_edit.py").is_file())
            self.assertTrue((record_dir / "post_edit.py").is_file())
            self.assertTrue((record_dir / "instruction.txt").is_file())
            self.assertTrue((record_dir / "instruction.json").is_file())

    def test_random_sampling_is_reproducible_and_balanced_by_type(self):
        records = []
        for index in range(20):
            instr_type = "descriptive" if index % 2 == 0 else "lazy"
            records.append(
                {
                    "commit": f"commit-{index}",
                    "code_before_purify": f"value = {index}\n",
                    "code_after_purify": f"value = {index + 1}\n",
                    "instruct_purify": f"Increment value {index}.",
                    "instr_type": instr_type,
                }
            )
        random_input = self.root / "random.jsonl"
        random_input.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

        first = export_first_samples(
            random_input,
            3,
            self.root / "random-first",
            instr_types=["descriptive", "lazy"],
            seed=42,
        )
        repeated = export_first_samples(
            random_input,
            3,
            self.root / "random-repeated",
            instr_types=["descriptive", "lazy"],
            seed=42,
        )
        different_seed = export_first_samples(
            random_input,
            3,
            self.root / "random-different",
            instr_types=["descriptive", "lazy"],
            seed=43,
        )

        relative_first = [
            path.relative_to(self.root / "random-first") for path in first
        ]
        relative_repeated = [
            path.relative_to(self.root / "random-repeated") for path in repeated
        ]
        relative_different = [
            path.relative_to(self.root / "random-different")
            for path in different_seed
        ]
        self.assertEqual(relative_first, relative_repeated)
        self.assertNotEqual(relative_first, relative_different)
        self.assertEqual(
            sum(path.parts[0] == "descriptive" for path in relative_first), 3
        )
        self.assertEqual(sum(path.parts[0] == "lazy" for path in relative_first), 3)

    def test_reports_type_with_fewer_than_k_records(self):
        with self.assertRaisesRegex(ValueError, "lazy: found 1/2"):
            export_first_samples(
                self.input_file,
                2,
                self.root / "by-type",
                instr_types=["lazy"],
            )

    def test_rejects_instr_type_that_would_create_nested_directories(self):
        with self.assertRaisesRegex(ValueError, "cannot be used"):
            export_first_samples(
                self.input_file,
                1,
                self.root / "by-type",
                instr_types=["../lazy"],
            )

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
        self.assertTrue(
            (output_dir / "line_000001" / "instruction.json").is_file()
        )


if __name__ == "__main__":
    unittest.main()
