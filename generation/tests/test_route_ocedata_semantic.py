import json
from pathlib import Path
import tempfile
import unittest

import yaml

from generation.route_ocedata_semantic import prepare, _parser, finalize


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


class RouteOCEDataSemanticTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.records = [
            {"commit": "a", "instr_type": "qwen3_lazy", "instruct_purify": "i1"},
            {"commit": "b", "instr_type": "ds_descriptive", "instruct_purify": "i2"},
            {"commit": "c", "instr_type": "qwen3_descriptive", "instruct_purify": "i3"},
        ]

    def tearDown(self):
        self.temporary.cleanup()

    def test_prepare_preserves_route_order(self):
        source = self.root / "quality.jsonl"
        qwen = self.root / "qwen.jsonl"
        ds = self.root / "ds.jsonl"
        write_jsonl(source, self.records)

        summary = prepare(source, qwen, ds)

        self.assertEqual(summary["routes"], {"ds": 1, "qwen3": 2})
        self.assertEqual(
            [json.loads(line)["commit"] for line in qwen.read_text().splitlines()],
            ["a", "c"],
        )
        self.assertEqual(json.loads(ds.read_text())["commit"], "b")

    def test_prepare_rejects_unknown_generator(self):
        source = self.root / "quality.jsonl"
        write_jsonl(source, [{"instr_type": "other_lazy"}])
        with self.assertRaisesRegex(ValueError, "unsupported instr_type"):
            prepare(source, self.root / "qwen.jsonl", self.root / "ds.jsonl")

    def test_finalize_restores_order_and_reuses_ft_decisions(self):
        main_source = self.root / "main.jsonl"
        main_quality = self.root / "main_quality.jsonl"
        ft_source = self.root / "ft.jsonl"
        ft_quality = self.root / "ft_quality.jsonl"
        qwen_input = self.root / "qwen.jsonl"
        ds_input = self.root / "ds.jsonl"
        qwen_results = self.root / "qwen_results.jsonl"
        ds_results = self.root / "ds_results.jsonl"
        write_jsonl(main_source, self.records)
        write_jsonl(main_quality, self.records)
        write_jsonl(ft_source, [self.records[2], self.records[1]])
        write_jsonl(ft_quality, [self.records[2], self.records[1]])
        prepare(main_quality, qwen_input, ds_input)
        write_jsonl(
            qwen_results,
            [
                {"line_number": 1, "decision": "ACCEPT"},
                {"line_number": 2, "decision": "REJECT"},
            ],
        )
        write_jsonl(ds_results, [{"line_number": 1, "decision": "ACCEPT"}])
        paths = {
            name: self.root / name
            for name in (
                "main_results",
                "main_filtered",
                "main_summary",
                "ft_results",
                "ft_filtered",
                "ft_summary",
            )
        }
        args = _parser().parse_args(
            [
                "finalize",
                "--main-source", str(main_source),
                "--main-quality", str(main_quality),
                "--ft-source", str(ft_source),
                "--ft-quality", str(ft_quality),
                "--qwen-input", str(qwen_input),
                "--ds-input", str(ds_input),
                "--qwen-results", str(qwen_results),
                "--ds-results", str(ds_results),
                "--main-results", str(paths["main_results"]),
                "--main-filtered", str(paths["main_filtered"]),
                "--main-summary", str(paths["main_summary"]),
                "--ft-results", str(paths["ft_results"]),
                "--ft-filtered", str(paths["ft_filtered"]),
                "--ft-summary", str(paths["ft_summary"]),
            ]
        )

        finalize(args)

        self.assertEqual(
            [json.loads(line)["commit"] for line in paths["main_filtered"].read_text().splitlines()],
            ["a", "b"],
        )
        self.assertEqual(
            [json.loads(line)["commit"] for line in paths["ft_filtered"].read_text().splitlines()],
            ["b"],
        )
        ft_results = [json.loads(line) for line in paths["ft_results"].read_text().splitlines()]
        self.assertTrue(all(item["semantic_inference_reused_from_ocedata"] for item in ft_results))
        summary = yaml.safe_load(paths["ft_summary"].read_text())
        self.assertEqual(summary["counts"]["final_passed"], 1)


if __name__ == "__main__":
    unittest.main()
