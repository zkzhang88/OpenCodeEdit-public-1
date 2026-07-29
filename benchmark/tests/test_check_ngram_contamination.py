import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from benchmark.check_ngram_contamination import (
    BenchmarkField,
    CODEEDITOR_FILES,
    SUBSET_ORDER,
    compute_metrics,
    exact_containment,
    load_codeeditor_file,
    run_detection,
    scan_commitpack,
    tokenize_code,
    unique_ngrams,
)


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_equal_code_has_all_three_maximal_metrics():
    tokens = tokenize_code("def add(a, b):\n    return a + b\n").tokens
    metrics = compute_metrics(tokens, tokens, ngram_size=3)
    assert metrics["containment"] == 1.0
    assert metrics["exact_containment"] is True
    assert metrics["longest_matched_span"] == len(tokens)
    assert metrics["longest_matched_span_ratio"] == 1.0


def test_exact_containment_does_not_require_whole_file_equality():
    benchmark = tokenize_code("def add(a, b): return a + b").tokens
    seed = tokenize_code(
        "x = 1\ndef add(a,b):  # ignored\n    return a+b\ny = 2"
    ).tokens
    assert benchmark != seed
    assert exact_containment(benchmark, seed)
    metrics = compute_metrics(benchmark, seed, ngram_size=3)
    assert metrics["containment"] == 1.0
    assert metrics["longest_matched_span"] == len(benchmark)


def test_partial_copy_reports_longest_contiguous_span():
    benchmark = ("a", "b", "c", "d", "e")
    seed = ("x", "b", "c", "d", "y")
    metrics = compute_metrics(benchmark, seed, ngram_size=2)
    assert metrics["exact_containment"] is False
    assert metrics["longest_matched_span"] == 3


def test_comments_whitespace_and_fences_are_ignored():
    left = tokenize_code("```python\ndef f(x):\n    return x + 1\n```").tokens
    right = tokenize_code("def f( x ): # note\n return x+1").tokens
    assert left == right


def test_unique_ngrams_do_not_count_repetition_twice():
    grams = unique_ngrams(("a", "b", "a", "b", "a", "b"), 2)
    assert grams == frozenset({("a", "b"), ("b", "a")})


def test_no_overlap_and_short_benchmark_behavior():
    no_overlap = compute_metrics(("a", "b", "c"), ("x", "y", "z"), 2)
    assert no_overlap["containment"] == 0.0
    short = compute_metrics(("a", "b"), ("x", "a", "b", "y"), 3)
    assert short["containment"] is None
    assert short["exact_containment"] is True


def test_scan_keeps_top_k_in_stable_metric_order(tmp_path):
    benchmark_tokens = tokenize_code(
        "def add(a, b): value = a + b; return value"
    ).tokens
    field = BenchmarkField(
        field_id=0,
        subset="canitedit_test",
        task_id="1",
        field_name="before",
        tokens=benchmark_tokens,
        lexer_fallback=False,
        ngrams=unique_ngrams(benchmark_tokens, 3),
    )
    commitpack = tmp_path / "commitpack.jsonl"
    _write_jsonl(
        commitpack,
        [
            {
                "commit": str(index),
                "old_contents": code,
                "repos": "r",
                "old_file": "f.py",
            }
            for index, code in enumerate(
                [
                    "def add(a, b): return a - b",
                    "def add(a, b): value = a + b; return value",
                    "prefix = 1\ndef add(a, b): value = a + b; return value",
                    "def add(a, b): value = a + b; return 0",
                ],
                1,
            )
        ],
    )
    candidates, scanned = scan_commitpack(commitpack, [field], 3, top_k=2)
    assert scanned == 4
    assert len(candidates[0]) == 2
    ranks = [candidate.rank() for candidate in candidates[0]]
    assert ranks == sorted(ranks, reverse=True)
    assert all(candidate.exact_containment for candidate in candidates[0])


def test_codeeditor_python_field_mapping(tmp_path):
    debug_path = tmp_path / "code_debug_primary.jsonl"
    _write_jsonl(
        debug_path,
        [
            {
                "idx": 1,
                "code_language": "python3",
                "incorrect_solutions": "def f(): return 0",
                "solutions": "def f(): return 1",
            },
            {
                "idx": 2,
                "code_language": "java",
                "incorrect_solutions": "class A {}",
                "solutions": "class A {}",
            },
        ],
    )
    fields = load_codeeditor_file(debug_path, ngram_size=3)
    assert [(field.task_id, field.field_name) for field in fields] == [
        ("1", "incorrect_solutions"),
        ("1", "solutions"),
    ]

    translate_path = tmp_path / "code_translate_plus.jsonl"
    _write_jsonl(
        translate_path,
        [
            {
                "idx": 3,
                "source_lang": "python",
                "target_lang": "java",
                "source_code": "def f(): return 1",
                "target_code": "int f() { return 1; }",
            },
            {
                "idx": 4,
                "source_lang": "java",
                "target_lang": "python",
                "source_code": "int g() { return 2; }",
                "target_code": "def g(): return 2",
            },
        ],
    )
    fields = load_codeeditor_file(translate_path, ngram_size=3)
    assert [(field.task_id, field.field_name) for field in fields] == [
        ("3", "source_code"),
        ("4", "target_code"),
    ]


def test_end_to_end_writes_nine_independent_subsets(tmp_path):
    commitpack = tmp_path / "commitpack.jsonl"
    shared = "def add(a, b):\n    value = a + b\n    return value\n"
    _write_jsonl(
        commitpack,
        [
            {
                "commit": "abc",
                "repos": "owner/repo",
                "old_file": "math.py",
                "old_contents": shared,
            },
            {
                "commit": "def",
                "repos": "owner/other",
                "old_file": "other.py",
                "old_contents": "def other(x): return x * 2",
            },
        ],
    )

    canitedit = tmp_path / "canitedit.parquet"
    pq.write_table(
        pa.table(
            {
                "id": [1],
                "before": [shared],
                "after": ["def add(a, b): return a - b"],
            }
        ),
        canitedit,
    )

    codeeditor_dir = tmp_path / "codeeditor"
    common_code = shared
    rows_by_file = {
        "code_debug_primary.jsonl": [{
            "idx": 10, "code_language": "python3",
            "incorrect_solutions": common_code, "solutions": common_code,
        }],
        "code_debug_plus.jsonl": [{
            "idx": 11, "code_language": "python3",
            "incorrect_solutions": common_code, "solutions": common_code,
        }],
        "code_polishment_primary.jsonl": [{
            "idx": 12, "source_lang": "python", "source_code": common_code,
        }],
        "code_polishment_plus.jsonl": [{
            "idx": 13, "source_lang": "python", "source_code": common_code,
        }],
        "code_switch_primary.jsonl": [{
            "idx": 14, "language": "python",
            "similar_source_code": common_code, "target_source_code": common_code,
        }],
        "code_switch_plus.jsonl": [{
            "idx": 15, "language": "python",
            "similar_source_code": common_code, "target_source_code": common_code,
        }],
        "code_translate_primary.jsonl": [{
            "idx": 16, "source_lang": "python", "target_lang": "java",
            "source_code": common_code, "target_code": "class A {}",
        }],
        "code_translate_plus.jsonl": [{
            "idx": 17, "source_lang": "java", "target_lang": "python",
            "source_code": "class A {}", "target_code": common_code,
        }],
    }
    assert set(rows_by_file) == set(CODEEDITOR_FILES)
    for filename, rows in rows_by_file.items():
        _write_jsonl(codeeditor_dir / filename, rows)

    output_dir = tmp_path / "results"
    index = run_detection(
        commitpack,
        canitedit,
        codeeditor_dir,
        output_dir,
        ngram_size=3,
        threshold=0.8,
        top_k=1,
    )

    assert [row["subset"] for row in index] == list(SUBSET_ORDER)
    assert all(row["total_tasks"] == 1 for row in index)
    assert all(row["contaminated_tasks"] == 1 for row in index)
    for subset in SUBSET_ORDER:
        assert (output_dir / subset / "matches.jsonl").is_file()
        assert (output_dir / subset / "task_summary.csv").is_file()
        assert (output_dir / subset / "summary.json").is_file()

    with (output_dir / "canitedit_test" / "summary.json").open(
        encoding="utf-8"
    ) as handle:
        summary = json.load(handle)
    assert summary["fields"]["before"]["contaminated_fields"] == 1
    assert summary["fields"]["after"]["total_fields"] == 1
    assert summary["scanned_commitpack_seeds"] == 2

    with (output_dir / "summary_index.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 9
