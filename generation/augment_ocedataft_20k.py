#!/usr/bin/env python3
"""Build the balanced 20K OCEData fine-tuning dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

if __package__:
    from generation.augment_ocedata_60k import (
        DATA_DIR,
        GENERATION_DATA_DIR,
        SELECTION_SEED,
        augment_dataset,
    )
else:  # Direct execution: python generation/<script>.py
    from augment_ocedata_60k import (  # type: ignore[no-redef]
        DATA_DIR,
        GENERATION_DATA_DIR,
        SELECTION_SEED,
        augment_dataset,
    )


DEFAULT_BASE = DATA_DIR / "ocedataft_quality_semantic_filtered.jsonl"
DEFAULT_CANDIDATES = (
    GENERATION_DATA_DIR / "filtered" / "ocedata_mix_all_descriptive_dt_filtered.jsonl",
    GENERATION_DATA_DIR / "filtered" / "ocedata_mix_all_lazy_dt_filtered.jsonl",
)
DEFAULT_OUTPUT = DATA_DIR / "ocedataft_quality_semantic_filtered_augmented_20k.jsonl"
DEFAULT_SUMMARY = (
    DATA_DIR / "ocedataft_quality_semantic_filtered_augmented_20k_summary.yaml"
)
TARGET_PER_STYLE = 10_000


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Augment semantic-filtered OCEData FT to balanced styles at 20K."
    )
    parser.add_argument("--base-file", type=Path, default=DEFAULT_BASE)
    parser.add_argument(
        "--candidate-file",
        type=Path,
        action="append",
        dest="candidate_files",
        help="Candidate JSONL; repeat for multiple files (defaults to DT-filtered files).",
    )
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-file", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--target-per-style", type=int, default=TARGET_PER_STYLE)
    parser.add_argument("--selection-seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = augment_dataset(
        args.base_file,
        args.candidate_files or list(DEFAULT_CANDIDATES),
        args.output_file,
        args.summary_file,
        target_per_style=args.target_per_style,
        selection_seed=args.selection_seed,
        overwrite=args.overwrite,
    )
    print(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
