import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from benchmark.check_ngram_contamination import (
    BenchmarkField,
    CODEEDITOR_FILES,
    DEFAULT_OUTPUT_DIR,
    SUBSET_ORDER,
    build_parser,
    compute_metrics,
    default_output_dir,
    exact_containment,
    load_codeeditor_file,
    run_detection,
    run_multi_detection,
    scan_commitpack,
    scan_seed_file,
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


def test_oneshot_scan_uses_only_code_before_and_tracks_source(tmp_path):
    benchmark_tokens = tokenize_code("def add(a, b): return a + b").tokens
    field = BenchmarkField(
        field_id=0,
        subset="canitedit_test",
        task_id="1",
        field_name="before",
        tokens=benchmark_tokens,
        lexer_fallback=False,
        ngrams=unique_ngrams(benchmark_tokens, 3),
    )
    oneshot = tmp_path / "oneshot.jsonl"
    _write_jsonl(
        oneshot,
        [
            {
                "code_before": "def unrelated(): return None",
                "code_after": "def add(a, b): return a + b",
                "code_after_purify": "def add(a, b): return a + b",
            },
            {
                "code_before": "def add(a, b): return a + b",
                "code_after": "def add(a, b): return a - b",
            },
        ],
    )

    candidates, scanned = scan_seed_file(
        oneshot, [field], 3, top_k=1, seed_source="oneshot"
    )

    assert scanned == 2
    assert len(candidates[0]) == 1
    seed = candidates[0][0].seed
    assert seed.source == "oneshot"
    assert seed.line_number == 2
    assert seed.field_name == "code_before"
    assert seed.commit == seed.repo == seed.old_file == ""


def test_seed_source_has_separate_default_output_directory():
    assert default_output_dir("commitpack") == DEFAULT_OUTPUT_DIR
    assert default_output_dir("oneshot") == DEFAULT_OUTPUT_DIR / "oneshot"


def test_cli_routes_oneshot_source_and_paths():
    args = build_parser().parse_args(
        [
            "--seed-source",
            "oneshot",
            "--oneshot",
            "custom-oneshot.jsonl",
            "--output-dir",
            "custom-results",
        ]
    )
    assert args.seed_source == "oneshot"
    assert args.oneshot == Path("custom-oneshot.jsonl")
    assert args.output_dir == Path("custom-results")


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

    oneshot = tmp_path / "oneshot.jsonl"
    _write_jsonl(
        oneshot,
        [{"code_before": shared, "code_after": "def unrelated(): pass"}],
    )
    oneshot_output = tmp_path / "oneshot-results"
    oneshot_index = run_detection(
        commitpack,
        canitedit,
        codeeditor_dir,
        oneshot_output,
        ngram_size=3,
        threshold=0.8,
        top_k=1,
        seed_source="oneshot",
        oneshot_path=oneshot,
    )
    assert all(row["contaminated_tasks"] == 1 for row in oneshot_index)

    with (oneshot_output / "canitedit_test" / "matches.jsonl").open(
        encoding="utf-8"
    ) as handle:
        matches = [json.loads(line) for line in handle]
    before_match = next(
        row for row in matches if row["benchmark_field"] == "before"
    )
    assert before_match["seed_source"] == "oneshot"
    assert before_match["seed_line"] == 1
    assert before_match["seed_field"] == "code_before"
    assert before_match["commitpack_line"] is None
    assert before_match["commit"] == before_match["repo"] == ""

    with (oneshot_output / "canitedit_test" / "summary.json").open(
        encoding="utf-8"
    ) as handle:
        oneshot_summary = json.load(handle)
    assert oneshot_summary["seed_source"] == "oneshot"
    assert oneshot_summary["scanned_seed_records"] == 1
    assert oneshot_summary["scanned_commitpack_seeds"] == 0
    assert oneshot_summary["scanned_oneshot_seeds"] == 1


def test_multi_detection_reports_each_size_separately(tmp_path):
    commitpack = tmp_path / "commitpack.jsonl"
    shared = "def add(a, b): value = a + b; return value"
    _write_jsonl(
        commitpack,
        [{
            "commit": "abc",
            "repos": "owner/repo",
            "old_file": "math.py",
            "old_contents": shared,
        }],
    )

    canitedit = tmp_path / "canitedit.parquet"
    pq.write_table(
        pa.table({"id": [1], "before": [shared], "after": [shared]}),
        canitedit,
    )

    codeeditor_dir = tmp_path / "codeeditor"
    rows_by_file = {
        "code_debug_primary.jsonl": [{
            "idx": 10, "code_language": "python3",
            "incorrect_solutions": shared, "solutions": shared,
        }],
        "code_debug_plus.jsonl": [{
            "idx": 11, "code_language": "python3",
            "incorrect_solutions": shared, "solutions": shared,
        }],
        "code_polishment_primary.jsonl": [{
            "idx": 12, "source_lang": "python", "source_code": shared,
        }],
        "code_polishment_plus.jsonl": [{
            "idx": 13, "source_lang": "python", "source_code": shared,
        }],
        "code_switch_primary.jsonl": [{
            "idx": 14, "language": "python",
            "similar_source_code": shared, "target_source_code": shared,
        }],
        "code_switch_plus.jsonl": [{
            "idx": 15, "language": "python",
            "similar_source_code": shared, "target_source_code": shared,
        }],
        "code_translate_primary.jsonl": [{
            "idx": 16, "source_lang": "python", "target_lang": "java",
            "source_code": shared, "target_code": "class A {}",
        }],
        "code_translate_plus.jsonl": [{
            "idx": 17, "source_lang": "java", "target_lang": "python",
            "source_code": "class A {}", "target_code": shared,
        }],
    }
    for filename, rows in rows_by_file.items():
        _write_jsonl(codeeditor_dir / filename, rows)

    output_dir = tmp_path / "results"
    index = run_multi_detection(
        commitpack,
        canitedit,
        codeeditor_dir,
        output_dir,
        ngram_sizes=[2, 4],
        threshold=0.8,
        top_k=1,
    )

    assert len(index) == 18
    assert {row["ngram_size"] for row in index} == {2, 4}
    assert (output_dir / "ngram_2" / "summary_index.csv").is_file()
    assert (output_dir / "ngram_4" / "summary_index.csv").is_file()
    with (output_dir / "summary_index.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 18
    assert {row["ngram_size"] for row in rows} == {"2", "4"}

    oneshot = tmp_path / "oneshot.jsonl"
    _write_jsonl(oneshot, [{"code_before": shared, "code_after": "ignored"}])
    oneshot_output = tmp_path / "oneshot-results"
    oneshot_index = run_multi_detection(
        commitpack,
        canitedit,
        codeeditor_dir,
        oneshot_output,
        ngram_sizes=[2, 4],
        threshold=0.8,
        top_k=1,
        seed_source="oneshot",
        oneshot_path=oneshot,
    )
    assert len(oneshot_index) == 18
    assert {row["ngram_size"] for row in oneshot_index} == {2, 4}
    assert (oneshot_output / "ngram_2" / "summary_index.csv").is_file()
    assert (oneshot_output / "ngram_4" / "summary_index.csv").is_file()
