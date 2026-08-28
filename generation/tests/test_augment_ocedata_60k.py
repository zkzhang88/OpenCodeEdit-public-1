import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from generation.augment_ocedata_60k import (
    INSTR_TYPES,
    OUTPUT_FIELDS,
    _output_record,
    _selection_rank,
    augment_dataset,
    main,
)
from generation.augment_ocedataft_20k import main as ft_main


def record(commit, instr_type, *, extra=False):
    value = {
        "commit": commit,
        "code_before_purify": f"before {commit}",
        "code_after_purify": f"after {commit}",
        "instruct_purify": f"edit {commit}",
        "instr_type": instr_type,
    }
    if extra:
        value["code_snippet"] = ["source metadata"]
    return value


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in records),
        encoding="utf-8",
    )


class AugmentOCEDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.base = self.root / "base.jsonl"
        self.descriptive = self.root / "descriptive.jsonl"
        self.lazy = self.root / "lazy.jsonl"
        self.output = self.root / "output.jsonl"
        self.summary = self.root / "summary.yaml"

        base_records = []
        candidate_files = {"descriptive": [], "lazy": []}
        for instr_type in INSTR_TYPES:
            base_records.append(record(f"base-{instr_type}", instr_type))
            style = "descriptive" if instr_type.endswith("descriptive") else "lazy"
            for index in range(5):
                candidate_files[style].append(
                    record(f"candidate-{instr_type}-{index}", instr_type, extra=True)
                )
        # Existing overlaps are excluded and do not affect the quotas.
        candidate_files["lazy"].append(record("base-ds_lazy", "ds_lazy", extra=True))
        write_jsonl(self.base, base_records)
        write_jsonl(self.descriptive, candidate_files["descriptive"])
        write_jsonl(self.lazy, candidate_files["lazy"])

    def tearDown(self):
        self.temporary.cleanup()

    def run_augmentation(self, *, output=None, summary=None, seed=42):
        return augment_dataset(
            self.base,
            [self.descriptive, self.lazy],
            output or self.output,
            summary or self.summary,
            target_per_type=3,
            selection_seed=seed,
        )

    def test_balances_four_types_preserves_base_and_normalizes_schema(self):
        base_bytes = self.base.read_bytes()
        summary = self.run_augmentation()

        output_bytes = self.output.read_bytes()
        self.assertTrue(output_bytes.startswith(base_bytes))
        records = [json.loads(line) for line in output_bytes.splitlines()]
        counts = {kind: 0 for kind in INSTR_TYPES}
        for value in records:
            counts[value["instr_type"]] += 1
            self.assertEqual(list(value), list(OUTPUT_FIELDS))
            self.assertNotIn("code_snippet", value)
        self.assertEqual(counts, {kind: 3 for kind in INSTR_TYPES})
        self.assertEqual(summary["counts"]["final_by_model"], {"ds": 6, "qwen3": 6})
        self.assertEqual(
            summary["counts"]["final_by_instruction_style"],
            {"descriptive": 6, "lazy": 6},
        )
        self.assertEqual(summary["counts"]["added_total"], 8)
        self.assertEqual(summary["candidate_audit"]["excluded_existing_commit"], 1)

        added_commits = [value["commit"] for value in records[4:]]
        self.assertEqual(len(added_commits), len(set(added_commits)))
        self.assertTrue(set(added_commits).isdisjoint(value["commit"] for value in records[:4]))
        persisted = yaml.safe_load(self.summary.read_text(encoding="utf-8"))
        self.assertEqual(
            persisted["output_sha256"], hashlib.sha256(output_bytes).hexdigest()
        )

    def test_global_commit_collision_has_one_deterministic_winner(self):
        collision = "shared-candidate"
        left = record(collision, "ds_descriptive", extra=True)
        right = record(collision, "qwen3_lazy", extra=True)
        expected = min(
            (left, right),
            key=lambda value: _selection_rank(_output_record(value), 42),
        )["instr_type"]
        candidates = {kind: [] for kind in INSTR_TYPES}
        candidates["ds_descriptive"].append(left)
        candidates["qwen3_lazy"].append(right)
        for kind in INSTR_TYPES:
            # The winning type gets one alternative and must select the
            # collision; the losing type has two alternatives after dedup.
            alternative_count = 1 if kind == expected else 2
            candidates[kind].extend(
                record(f"alternative-{kind}-{index}", kind, extra=True)
                for index in range(alternative_count)
            )
        write_jsonl(
            self.descriptive,
            candidates["ds_descriptive"] + candidates["qwen3_descriptive"],
        )
        write_jsonl(
            self.lazy,
            candidates["ds_lazy"] + candidates["qwen3_lazy"],
        )

        summary = self.run_augmentation()
        values = [json.loads(line) for line in self.output.read_text().splitlines()]
        collision_values = [value for value in values if value["commit"] == collision]
        self.assertEqual([value["instr_type"] for value in collision_values], [expected])
        self.assertEqual(
            summary["candidate_audit"]["duplicate_candidate_commit_records"], 1
        )

    def test_reproducible_and_seed_can_change_selection(self):
        first_summary = self.run_augmentation()
        first = self.output.read_bytes()
        second_output = self.root / "second.jsonl"
        second_summary_path = self.root / "second.yaml"
        second_summary = self.run_augmentation(
            output=second_output, summary=second_summary_path
        )
        third_output = self.root / "third.jsonl"
        third_summary_path = self.root / "third.yaml"
        self.run_augmentation(
            output=third_output, summary=third_summary_path, seed=7
        )

        self.assertEqual(first, second_output.read_bytes())
        self.assertEqual(first_summary["output_sha256"], second_summary["output_sha256"])
        self.assertNotEqual(first, third_output.read_bytes())

    def test_insufficient_pool_leaves_no_outputs(self):
        write_jsonl(self.lazy, [record("only-one", "ds_lazy", extra=True)])
        with self.assertRaisesRegex(ValueError, "ds_lazy: found 1/2"):
            self.run_augmentation()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.summary.exists())

    def test_rejects_existing_output_without_changing_it(self):
        self.output.write_text("keep me\n", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            self.run_augmentation()
        self.assertEqual(self.output.read_text(), "keep me\n")
        self.assertFalse(self.summary.exists())

    def test_balances_styles_without_model_quota(self):
        write_jsonl(
            self.descriptive,
            [record(f"descriptive-ds-{index}", "ds_descriptive", extra=True)
             for index in range(3)],
        )
        write_jsonl(
            self.lazy,
            [record(f"lazy-ds-{index}", "ds_lazy", extra=True)
             for index in range(3)],
        )

        summary = augment_dataset(
            self.base,
            [self.descriptive, self.lazy],
            self.output,
            self.summary,
            target_per_style=3,
        )

        self.assertEqual(
            summary["counts"]["final_by_instruction_style"],
            {"descriptive": 3, "lazy": 3},
        )
        self.assertEqual(summary["counts"]["final_by_model"], {"ds": 4, "qwen3": 2})
        self.assertEqual(summary["balance"]["by"], "instruction_style")

    def test_ft_cli_accepts_overrides(self):
        result = ft_main(
            [
                "--base-file", str(self.base),
                "--candidate-file", str(self.descriptive),
                "--candidate-file", str(self.lazy),
                "--output-file", str(self.output),
                "--summary-file", str(self.summary),
                "--target-per-style", "3",
            ]
        )
        self.assertEqual(result, 0)
        persisted = yaml.safe_load(self.summary.read_text(encoding="utf-8"))
        self.assertEqual(persisted["counts"]["final_total"], 6)

    def test_cli_accepts_overrides(self):
        result = main(
            [
                "--base-file", str(self.base),
                "--candidate-file", str(self.descriptive),
                "--candidate-file", str(self.lazy),
                "--output-file", str(self.output),
                "--summary-file", str(self.summary),
                "--target-per-type", "3",
            ]
        )
        self.assertEqual(result, 0)
        self.assertTrue(self.output.is_file())
        self.assertTrue(self.summary.is_file())


if __name__ == "__main__":
    unittest.main()
