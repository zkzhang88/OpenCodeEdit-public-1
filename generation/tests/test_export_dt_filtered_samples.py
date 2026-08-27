import json
from pathlib import Path
import tempfile
import unittest

from generation.export_dt_filtered_samples import (
    INSTR_TYPES,
    MANUAL_REVIEW_FIELD,
    main,
    sample_and_export,
    sample_records,
)


class ExportDtFilteredSamplesTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.descriptive_file = self.root / "descriptive_dt_filtered.jsonl"
        self.lazy_file = self.root / "lazy_dt_filtered.jsonl"
        self.files = [self.descriptive_file, self.lazy_file]
        self.records = {}

        for instr_type in INSTR_TYPES:
            records = []
            for index in range(1, 9):
                records.append(
                    {
                        "commit": f"{instr_type}-commit-{index}",
                        "code_before_purify": f"value = {index}\n",
                        "code_after_purify": f"value = {index + 1}\n",
                        "instruct_purify": f"Update {instr_type} sample {index}.",
                        "instr_type": instr_type,
                    }
                )
            self.records[instr_type] = records

        self._write_records(
            self.descriptive_file,
            self.records["ds_descriptive"]
            + self.records["qwen3_descriptive"],
        )
        self._write_records(
            self.lazy_file,
            self.records["ds_lazy"] + self.records["qwen3_lazy"],
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def _write_records(path, records):
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    @staticmethod
    def _selected_commits(samples):
        return {
            instr_type: [item.record["commit"] for item in records]
            for instr_type, records in samples.items()
        }

    def test_samples_equal_number_from_each_type(self):
        samples = sample_records(self.files, count=16, seed=42)

        self.assertEqual(list(samples), list(INSTR_TYPES))
        self.assertEqual(
            {instr_type: len(records) for instr_type, records in samples.items()},
            {instr_type: 4 for instr_type in INSTR_TYPES},
        )
        for records in samples.values():
            locations = [
                (str(record.source_file), record.source_line_number)
                for record in records
            ]
            self.assertEqual(locations, sorted(locations))

    def test_seed_is_reproducible_and_can_change_sample(self):
        first = self._selected_commits(
            sample_records(self.files, count=16, seed=42)
        )
        repeated = self._selected_commits(
            sample_records(self.files, count=16, seed=42)
        )
        different = self._selected_commits(
            sample_records(self.files, count=16, seed=7)
        )

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, different)

    def test_exports_code_metadata_and_only_one_manual_field(self):
        output_dir = self.root / "exports"
        exported = sample_and_export(
            self.files, output_dir, count=4, seed=42
        )

        self.assertEqual(len(exported), 4)
        self.assertEqual(
            [directory.parent.name for directory in exported],
            list(INSTR_TYPES),
        )
        for directory in exported:
            self.assertTrue((directory / "pre_edit.py").is_file())
            self.assertTrue((directory / "post_edit.py").is_file())
            metadata = json.loads(
                (directory / "instruction.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata[MANUAL_REVIEW_FIELD], None)
            self.assertEqual(
                [key for key in metadata if key.startswith("manual_")],
                [MANUAL_REVIEW_FIELD],
            )
            self.assertIn(metadata["source_file"], map(str, self.files))
            self.assertGreater(metadata["source_line_number"], 0)
            source_records = self.records[metadata["instr_type"]]
            expected = next(
                record
                for record in source_records
                if record["commit"] == metadata["commit"]
            )
            self.assertEqual(
                (directory / "pre_edit.py").read_text(encoding="utf-8"),
                expected["code_before_purify"],
            )
            self.assertEqual(
                (directory / "post_edit.py").read_text(encoding="utf-8"),
                expected["code_after_purify"],
            )

    def test_rejects_invalid_counts_before_creating_output(self):
        for count, message in ((0, "positive integer"), (6, "divisible by 4")):
            output_dir = self.root / f"invalid-{count}"
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, message):
                    sample_and_export(self.files, output_dir, count=count)
                self.assertFalse(output_dir.exists())

    def test_reports_missing_or_insufficient_type_without_output(self):
        output_dir = self.root / "incomplete-output"
        self._write_records(
            self.lazy_file,
            self.records["ds_lazy"] + self.records["qwen3_lazy"][:1],
        )

        with self.assertRaisesRegex(
            ValueError, "qwen3_lazy: found 1/2"
        ):
            sample_and_export(self.files, output_dir, count=8)
        self.assertFalse(output_dir.exists())

    def test_rejects_invalid_json_without_output(self):
        output_dir = self.root / "invalid-json-output"
        self.lazy_file.write_text("not-json\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Invalid JSON"):
            sample_and_export(self.files, output_dir, count=4)
        self.assertFalse(output_dir.exists())

    def test_rejects_missing_and_non_string_fields(self):
        cases = (
            ({"code_after_purify": None}, "must be a string"),
            ({"commit": None}, "must be a string"),
        )
        for index, (replacement, message) in enumerate(cases):
            with self.subTest(replacement=replacement):
                record = dict(self.records["ds_descriptive"][0])
                field = next(iter(replacement))
                if index == 0:
                    record.update(replacement)
                else:
                    record.pop(field)
                self._write_records(
                    self.descriptive_file,
                    [record]
                    + self.records["ds_descriptive"][1:]
                    + self.records["qwen3_descriptive"],
                )
                expected = message if index == 0 else "missing required field"
                with self.assertRaisesRegex(ValueError, expected):
                    sample_records(self.files, count=4)

    def test_rejects_unsupported_instruction_type(self):
        record = dict(self.records["ds_descriptive"][0])
        record["instr_type"] = "unknown"
        self._write_records(self.descriptive_file, [record])

        with self.assertRaisesRegex(ValueError, "unsupported instr_type"):
            sample_records(self.files, count=4)

    def test_cli_supports_parameter_overrides(self):
        output_dir = self.root / "cli-output"
        result = main(
            [
                "--input-file",
                *(str(path) for path in self.files),
                "--output-dir",
                str(output_dir),
                "--count",
                "8",
                "--seed",
                "7",
            ]
        )

        self.assertEqual(result, 0)
        for instr_type in INSTR_TYPES:
            self.assertEqual(
                len(list((output_dir / instr_type).glob("line_*"))), 2
            )


if __name__ == "__main__":
    unittest.main()
