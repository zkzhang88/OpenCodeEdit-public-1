"""Detect token n-gram overlap between seed data and code-edit benchmarks.

The seed data can be CommitPackFT ``old_contents`` or the ``code_before`` field
from the generation one-shot examples.  It is compared with Python code fields
from CanItEdit and each CodeEditorBench subset.  Results are reported
independently for every subset; no aggregate CodeEditorBench contamination rate
is computed.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import heapq
import io
import json
import math
import re
import statistics
import token
import tokenize
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple


DEFAULT_COMMITPACK = Path("generation/data/commitpackft_python_cleaned.jsonl")
DEFAULT_ONESHOT = Path("generation/few-shot/1-shot-prompt_final_chose.jsonl")
DEFAULT_CANITEDIT = Path(
    "benchmark/data/CanItEdit/test-00000-of-00001.parquet"
)
DEFAULT_CODEEDITOR_DIR = Path("benchmark/data/CodeEditorBench")
DEFAULT_OUTPUT_DIR = Path("benchmark/data/contamination_results")
DEFAULT_NGRAM_SIZES = (5, 10, 20)

CODEEDITOR_FILES = (
    "code_debug_primary.jsonl",
    "code_debug_plus.jsonl",
    "code_polishment_primary.jsonl",
    "code_polishment_plus.jsonl",
    "code_switch_primary.jsonl",
    "code_switch_plus.jsonl",
    "code_translate_primary.jsonl",
    "code_translate_plus.jsonl",
)

SUBSET_ORDER = (
    "canitedit_test",
    "code_debug_primary",
    "code_debug_plus",
    "code_polishment_primary",
    "code_polishment_plus",
    "code_switch_primary",
    "code_switch_plus",
    "code_translate_primary",
    "code_translate_plus",
)

_FENCE_RE = re.compile(r"^\s*```[^\n]*$", re.MULTILINE)
_SKIPPED_TOKEN_TYPES = {
    tokenize.ENCODING,
    tokenize.ENDMARKER,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.NEWLINE,
    tokenize.NL,
    tokenize.COMMENT,
}


@dataclass
class TokenizedCode:
    tokens: Tuple[str, ...]
    fallback: bool = False


@dataclass
class BenchmarkField:
    field_id: int
    subset: str
    task_id: str
    field_name: str
    tokens: Tuple[str, ...]
    lexer_fallback: bool
    ngrams: frozenset[Tuple[str, ...]] = field(default_factory=frozenset)


@dataclass
class SeedRecord:
    source: str
    line_number: int
    field_name: str
    commit: str
    repo: str
    old_file: str
    tokens: Tuple[str, ...]
    lexer_fallback: bool


@dataclass
class Candidate:
    field_id: int
    seed: SeedRecord
    containment: Optional[float]
    shared_ngram_count: int
    seed_ngram_count: int
    exact_containment: bool
    longest_matched_span: int

    def rank(self) -> Tuple[float, int, int, int]:
        containment = self.containment if self.containment is not None else 0.0
        return (
            containment,
            int(self.exact_containment),
            self.longest_matched_span,
            -self.seed.line_number,
        )


def _strip_markdown_fences(code: str) -> str:
    return _FENCE_RE.sub("", code)


def _fallback_tokenize(code: str) -> Tuple[str, ...]:
    try:
        from pygments import lex
        from pygments.lexers import PythonLexer
        from pygments.token import Comment
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "Pygments is required for tolerant tokenization. "
            "Install benchmark/requirements.txt."
        ) from exc

    values: List[str] = []
    for token_type, value in lex(code, PythonLexer()):
        if token_type in Comment or not value or value.isspace():
            continue
        values.append(value)
    return tuple(values)


def tokenize_code(code: object) -> TokenizedCode:
    """Canonicalize Python code into lexical tokens.

    Comments, formatting-only whitespace, newlines and Markdown fences are
    ignored. Identifiers, literals, keywords, and operators retain their text.
    Invalid Python falls back to Pygments' tolerant lexer.
    """

    text = _strip_markdown_fences(code if isinstance(code, str) else "")
    try:
        generated = tuple(tokenize.generate_tokens(io.StringIO(text).readline))
        if any(
            tok.type == token.ERRORTOKEN
            and tok.string
            and not tok.string.isspace()
            for tok in generated
        ):
            raise tokenize.TokenError("invalid token", (0, 0))
        values = tuple(
            tok.string
            for tok in generated
            if tok.type not in _SKIPPED_TOKEN_TYPES and tok.string
        )
        return TokenizedCode(values, fallback=False)
    except (IndentationError, SyntaxError, tokenize.TokenError):
        return TokenizedCode(_fallback_tokenize(text), fallback=True)


def unique_ngrams(
    tokens: Sequence[str], ngram_size: int
) -> frozenset[Tuple[str, ...]]:
    if ngram_size <= 0:
        raise ValueError("ngram_size must be positive")
    if len(tokens) < ngram_size:
        return frozenset()
    return frozenset(
        tuple(tokens[index : index + ngram_size])
        for index in range(len(tokens) - ngram_size + 1)
    )


def ngram_containment(
    benchmark_ngrams: frozenset[Tuple[str, ...]],
    seed_ngrams: frozenset[Tuple[str, ...]],
) -> Optional[float]:
    if not benchmark_ngrams:
        return None
    return len(benchmark_ngrams & seed_ngrams) / len(benchmark_ngrams)


def exact_containment(
    benchmark_tokens: Sequence[str], seed_tokens: Sequence[str]
) -> bool:
    """Return whether benchmark_tokens occur contiguously in seed_tokens."""

    if not benchmark_tokens:
        return False
    needle_length = len(benchmark_tokens)
    if needle_length > len(seed_tokens):
        return False

    # KMP prefix table keeps the check linear for large source files.
    prefix = [0] * needle_length
    matched = 0
    for index in range(1, needle_length):
        while matched and benchmark_tokens[index] != benchmark_tokens[matched]:
            matched = prefix[matched - 1]
        if benchmark_tokens[index] == benchmark_tokens[matched]:
            matched += 1
            prefix[index] = matched

    matched = 0
    for value in seed_tokens:
        while matched and value != benchmark_tokens[matched]:
            matched = prefix[matched - 1]
        if value == benchmark_tokens[matched]:
            matched += 1
            if matched == needle_length:
                return True
    return False


def longest_matched_span(
    benchmark_tokens: Sequence[str], seed_tokens: Sequence[str]
) -> int:
    """Return the longest contiguous common token span."""

    if not benchmark_tokens or not seed_tokens:
        return 0
    matcher = difflib.SequenceMatcher(
        None, benchmark_tokens, seed_tokens, autojunk=False
    )
    return matcher.find_longest_match(
        0, len(benchmark_tokens), 0, len(seed_tokens)
    ).size


def compute_metrics(
    benchmark_tokens: Sequence[str],
    seed_tokens: Sequence[str],
    ngram_size: int,
) -> Dict[str, object]:
    benchmark_ngrams = unique_ngrams(benchmark_tokens, ngram_size)
    seed_ngrams = unique_ngrams(seed_tokens, ngram_size)
    shared = len(benchmark_ngrams & seed_ngrams)
    span = longest_matched_span(benchmark_tokens, seed_tokens)
    return {
        "containment": (
            shared / len(benchmark_ngrams) if benchmark_ngrams else None
        ),
        "shared_ngram_count": shared,
        "benchmark_ngram_count": len(benchmark_ngrams),
        "seed_ngram_count": len(seed_ngrams),
        "exact_containment": exact_containment(
            benchmark_tokens, seed_tokens
        ),
        "longest_matched_span": span,
        "longest_matched_span_ratio": (
            span / len(benchmark_tokens) if benchmark_tokens else 0.0
        ),
    }


def _normalize_language(value: object) -> str:
    return str(value or "").strip().lower().replace("+", "p")


def _is_python(value: object) -> bool:
    return _normalize_language(value) in {"python", "python3", "py"}


def _add_benchmark_field(
    fields: List[BenchmarkField],
    subset: str,
    task_id: object,
    field_name: str,
    code: object,
    ngram_size: int,
) -> None:
    tokenized = tokenize_code(code)
    fields.append(
        BenchmarkField(
            field_id=len(fields),
            subset=subset,
            task_id=str(task_id),
            field_name=field_name,
            tokens=tokenized.tokens,
            lexer_fallback=tokenized.fallback,
            ngrams=unique_ngrams(tokenized.tokens, ngram_size),
        )
    )


def load_canitedit_fields(
    parquet_path: Path, ngram_size: int
) -> List[BenchmarkField]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "PyArrow is required to read CanItEdit. "
            "Install benchmark/requirements.txt."
        ) from exc

    table = pq.read_table(parquet_path, columns=["id", "before", "after"])
    data = table.to_pydict()
    fields: List[BenchmarkField] = []
    for task_id, before, after in zip(
        data["id"], data["before"], data["after"]
    ):
        _add_benchmark_field(
            fields, "canitedit_test", task_id, "before", before, ngram_size
        )
        _add_benchmark_field(
            fields, "canitedit_test", task_id, "after", after, ngram_size
        )
    return fields


def _iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, object]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if raw_line.strip():
                yield line_number, json.loads(raw_line)


def load_codeeditor_file(
    path: Path, ngram_size: int
) -> List[BenchmarkField]:
    subset = path.stem
    fields: List[BenchmarkField] = []
    for line_number, row in _iter_jsonl(path):
        task_id = row.get("idx", row.get("num", line_number))
        if subset.startswith("code_debug_"):
            if _is_python(row.get("code_language")):
                for field_name in ("incorrect_solutions", "solutions"):
                    _add_benchmark_field(
                        fields,
                        subset,
                        task_id,
                        field_name,
                        row.get(field_name, ""),
                        ngram_size,
                    )
        elif subset.startswith("code_polishment_"):
            if _is_python(row.get("source_lang")):
                _add_benchmark_field(
                    fields,
                    subset,
                    task_id,
                    "source_code",
                    row.get("source_code", ""),
                    ngram_size,
                )
        elif subset.startswith("code_switch_"):
            if _is_python(row.get("language")):
                for field_name in ("similar_source_code", "target_source_code"):
                    _add_benchmark_field(
                        fields,
                        subset,
                        task_id,
                        field_name,
                        row.get(field_name, ""),
                        ngram_size,
                    )
        elif subset.startswith("code_translate_"):
            if _is_python(row.get("source_lang")):
                _add_benchmark_field(
                    fields,
                    subset,
                    task_id,
                    "source_code",
                    row.get("source_code", ""),
                    ngram_size,
                )
            if _is_python(row.get("target_lang")):
                _add_benchmark_field(
                    fields,
                    subset,
                    task_id,
                    "target_code",
                    row.get("target_code", ""),
                    ngram_size,
                )
    return fields


def load_all_benchmark_fields(
    canitedit_path: Path, codeeditor_dir: Path, ngram_size: int
) -> List[BenchmarkField]:
    fields = load_canitedit_fields(canitedit_path, ngram_size)
    for filename in CODEEDITOR_FILES:
        path = codeeditor_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing CodeEditorBench subset: {path}")
        loaded = load_codeeditor_file(path, ngram_size)
        offset = len(fields)
        for local_id, item in enumerate(loaded):
            item.field_id = offset + local_id
        fields.extend(loaded)
    return fields


def _seed_from_row(
    line_number: int, row: Dict[str, object], source: str
) -> SeedRecord:
    if source == "commitpack":
        field_name = "old_contents"
        code = row.get(field_name, "")
    elif source == "oneshot":
        field_name = "code_before"
        code = row.get(field_name, "")
    else:
        raise ValueError(f"Unsupported seed source: {source}")

    tokenized = tokenize_code(code)
    return SeedRecord(
        source=source,
        line_number=line_number,
        field_name=field_name,
        commit=str(row.get("commit", "")) if source == "commitpack" else "",
        repo=str(row.get("repos", "")) if source == "commitpack" else "",
        old_file=(
            str(row.get("old_file", "")) if source == "commitpack" else ""
        ),
        tokens=tokenized.tokens,
        lexer_fallback=tokenized.fallback,
    )


def _candidate_for_pair(
    benchmark: BenchmarkField,
    seed: SeedRecord,
    seed_ngrams: frozenset[Tuple[str, ...]],
    shared_count: int,
) -> Candidate:
    span = longest_matched_span(benchmark.tokens, seed.tokens)
    return Candidate(
        field_id=benchmark.field_id,
        seed=seed,
        containment=(
            shared_count / len(benchmark.ngrams)
            if benchmark.ngrams
            else None
        ),
        shared_ngram_count=shared_count,
        seed_ngram_count=len(seed_ngrams),
        exact_containment=exact_containment(benchmark.tokens, seed.tokens),
        longest_matched_span=span,
    )


def scan_seed_file(
    seed_path: Path,
    fields: Sequence[BenchmarkField],
    ngram_size: int,
    top_k: int,
    seed_source: str,
) -> Tuple[Dict[int, List[Candidate]], int]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if seed_source not in {"commitpack", "oneshot"}:
        raise ValueError(f"Unsupported seed source: {seed_source}")

    inverted: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    short_fields: List[int] = []
    for benchmark in fields:
        if benchmark.ngrams:
            for ngram in benchmark.ngrams:
                inverted[ngram].append(benchmark.field_id)
        else:
            short_fields.append(benchmark.field_id)

    heaps: Dict[int, List[Tuple[Tuple[float, int, int, int], Candidate]]] = {
        benchmark.field_id: [] for benchmark in fields
    }
    field_by_id = {benchmark.field_id: benchmark for benchmark in fields}
    scanned = 0

    for line_number, row in _iter_jsonl(seed_path):
        scanned += 1
        seed = _seed_from_row(line_number, row, seed_source)
        seed_ngrams = unique_ngrams(seed.tokens, ngram_size)
        shared_counts: Counter[int] = Counter()
        for ngram in seed_ngrams:
            for field_id in inverted.get(ngram, ()):
                shared_counts[field_id] += 1

        for field_id, shared_count in shared_counts.items():
            benchmark = field_by_id[field_id]
            containment = shared_count / len(benchmark.ngrams)
            heap = heaps[field_id]
            if len(heap) >= top_k and containment < heap[0][0][0]:
                continue
            candidate = _candidate_for_pair(
                benchmark, seed, seed_ngrams, shared_count
            )
            item = (candidate.rank(), candidate)
            if len(heap) < top_k:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)

        # Short benchmark fields have no n-grams, so only exact containment can
        # make them contaminated under the documented rule.
        for field_id in short_fields:
            benchmark = field_by_id[field_id]
            if not exact_containment(benchmark.tokens, seed.tokens):
                continue
            candidate = _candidate_for_pair(
                benchmark, seed, seed_ngrams, shared_count=0
            )
            item = (candidate.rank(), candidate)
            heap = heaps[field_id]
            if len(heap) < top_k:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)

    ranked = {
        field_id: [
            candidate
            for _, candidate in sorted(heap, key=lambda item: item[0], reverse=True)
        ]
        for field_id, heap in heaps.items()
    }
    return ranked, scanned


def scan_commitpack(
    commitpack_path: Path,
    fields: Sequence[BenchmarkField],
    ngram_size: int,
    top_k: int,
) -> Tuple[Dict[int, List[Candidate]], int]:
    """Compatibility wrapper for scanning CommitPackFT seeds."""

    return scan_seed_file(
        commitpack_path, fields, ngram_size, top_k, "commitpack"
    )


def _is_contaminated(candidate: Optional[Candidate], threshold: float) -> bool:
    if candidate is None:
        return False
    return candidate.exact_containment or (
        candidate.containment is not None
        and candidate.containment >= threshold
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _score_distribution(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def write_results(
    fields: Sequence[BenchmarkField],
    candidates: Dict[int, List[Candidate]],
    output_dir: Path,
    ngram_size: int,
    threshold: float,
    top_k: int,
    scanned_seeds: int,
    seed_source: str = "commitpack",
) -> List[Dict[str, object]]:
    fields_by_subset: Dict[str, List[BenchmarkField]] = defaultdict(list)
    for benchmark in fields:
        fields_by_subset[benchmark.subset].append(benchmark)

    index_rows: List[Dict[str, object]] = []
    for subset in SUBSET_ORDER:
        subset_fields = fields_by_subset.get(subset, [])
        subset_dir = output_dir / subset
        subset_dir.mkdir(parents=True, exist_ok=True)

        with (subset_dir / "matches.jsonl").open("w", encoding="utf-8") as handle:
            for benchmark in subset_fields:
                for rank, candidate in enumerate(
                    candidates.get(benchmark.field_id, []), 1
                ):
                    span_ratio = (
                        candidate.longest_matched_span / len(benchmark.tokens)
                        if benchmark.tokens
                        else 0.0
                    )
                    record = {
                        "subset": subset,
                        "task_id": benchmark.task_id,
                        "benchmark_field": benchmark.field_name,
                        "rank": rank,
                        "seed_source": candidate.seed.source,
                        "seed_line": candidate.seed.line_number,
                        "seed_field": candidate.seed.field_name,
                        "commitpack_line": (
                            candidate.seed.line_number
                            if candidate.seed.source == "commitpack"
                            else None
                        ),
                        "commit": candidate.seed.commit,
                        "repo": candidate.seed.repo,
                        "old_file": candidate.seed.old_file,
                        "containment": candidate.containment,
                        "exact_containment": candidate.exact_containment,
                        "longest_matched_span": candidate.longest_matched_span,
                        "longest_matched_span_ratio": span_ratio,
                        "shared_ngram_count": candidate.shared_ngram_count,
                        "benchmark_ngram_count": len(benchmark.ngrams),
                        "seed_ngram_count": candidate.seed_ngram_count,
                        "benchmark_token_count": len(benchmark.tokens),
                        "seed_token_count": len(candidate.seed.tokens),
                        "benchmark_lexer_fallback": benchmark.lexer_fallback,
                        "seed_lexer_fallback": candidate.seed.lexer_fallback,
                        "contaminated": _is_contaminated(candidate, threshold),
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        tasks: Dict[str, List[BenchmarkField]] = defaultdict(list)
        for benchmark in subset_fields:
            tasks[benchmark.task_id].append(benchmark)

        task_rows: List[Dict[str, object]] = []
        for task_id, task_fields in sorted(tasks.items(), key=lambda item: item[0]):
            best_by_field = {
                item.field_name: (
                    candidates.get(item.field_id, [None])[0]
                    if candidates.get(item.field_id)
                    else None
                )
                for item in task_fields
            }
            contaminated_fields = sorted(
                field_name
                for field_name, best in best_by_field.items()
                if _is_contaminated(best, threshold)
            )
            containment_values = [
                best.containment
                for best in best_by_field.values()
                if best is not None and best.containment is not None
            ]
            task_rows.append(
                {
                    "task_id": task_id,
                    "field_count": len(task_fields),
                    "max_containment": (
                        max(containment_values) if containment_values else ""
                    ),
                    "any_exact_containment": any(
                        best is not None and best.exact_containment
                        for best in best_by_field.values()
                    ),
                    "max_longest_matched_span": max(
                        (
                            best.longest_matched_span
                            if best is not None
                            else 0
                        )
                        for best in best_by_field.values()
                    ),
                    "contaminated": bool(contaminated_fields),
                    "contaminated_fields": ";".join(contaminated_fields),
                }
            )

        task_columns = [
            "task_id",
            "field_count",
            "max_containment",
            "any_exact_containment",
            "max_longest_matched_span",
            "contaminated",
            "contaminated_fields",
        ]
        with (subset_dir / "task_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=task_columns)
            writer.writeheader()
            writer.writerows(task_rows)

        field_summaries: Dict[str, Dict[str, object]] = {}
        for field_name in sorted({item.field_name for item in subset_fields}):
            named_fields = [
                item for item in subset_fields if item.field_name == field_name
            ]
            best_candidates = [
                candidates.get(item.field_id, [None])[0]
                if candidates.get(item.field_id)
                else None
                for item in named_fields
            ]
            contaminated_count = sum(
                _is_contaminated(best, threshold) for best in best_candidates
            )
            scores = [
                best.containment if best and best.containment is not None else 0.0
                for best in best_candidates
            ]
            field_summaries[field_name] = {
                "total_fields": len(named_fields),
                "contaminated_fields": contaminated_count,
                "contamination_rate": (
                    contaminated_count / len(named_fields) if named_fields else 0.0
                ),
                "exact_containment_fields": sum(
                    bool(best and best.exact_containment)
                    for best in best_candidates
                ),
                "max_containment_distribution": _score_distribution(scores),
            }

        contaminated_tasks = sum(bool(row["contaminated"]) for row in task_rows)
        total_tasks = len(task_rows)
        task_scores = [
            float(row["max_containment"])
            if row["max_containment"] != ""
            else 0.0
            for row in task_rows
        ]
        summary = {
            "subset": subset,
            "total_tasks": total_tasks,
            "contaminated_tasks": contaminated_tasks,
            "contamination_rate": (
                contaminated_tasks / total_tasks if total_tasks else 0.0
            ),
            "total_code_fields": len(subset_fields),
            "seed_source": seed_source,
            "scanned_seed_records": scanned_seeds,
            "scanned_commitpack_seeds": (
                scanned_seeds if seed_source == "commitpack" else 0
            ),
            "scanned_oneshot_seeds": (
                scanned_seeds if seed_source == "oneshot" else 0
            ),
            "ngram_size": ngram_size,
            "threshold": threshold,
            "top_k": top_k,
            "task_max_containment_distribution": _score_distribution(task_scores),
            "fields": field_summaries,
        }
        with (subset_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        index_rows.append(
            {
                "subset": subset,
                "total_tasks": total_tasks,
                "contaminated_tasks": contaminated_tasks,
                "contamination_rate": summary["contamination_rate"],
                "total_code_fields": len(subset_fields),
                "max_containment": summary[
                    "task_max_containment_distribution"
                ]["max"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    index_columns = [
        "subset",
        "total_tasks",
        "contaminated_tasks",
        "contamination_rate",
        "total_code_fields",
        "max_containment",
    ]
    with (output_dir / "summary_index.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=index_columns)
        writer.writeheader()
        writer.writerows(index_rows)
    return index_rows


def run_detection(
    commitpack_path: Path,
    canitedit_path: Path,
    codeeditor_dir: Path,
    output_dir: Path,
    ngram_size: int = 10,
    threshold: float = 0.8,
    top_k: int = 5,
    seed_source: str = "commitpack",
    oneshot_path: Path = DEFAULT_ONESHOT,
) -> List[Dict[str, object]]:
    fields = load_all_benchmark_fields(
        canitedit_path, codeeditor_dir, ngram_size
    )
    seed_path = commitpack_path if seed_source == "commitpack" else oneshot_path
    candidates, scanned = scan_seed_file(
        seed_path, fields, ngram_size, top_k, seed_source
    )
    return write_results(
        fields,
        candidates,
        output_dir,
        ngram_size,
        threshold,
        top_k,
        scanned,
        seed_source,
    )


def run_multi_detection(
    commitpack_path: Path,
    canitedit_path: Path,
    codeeditor_dir: Path,
    output_dir: Path,
    ngram_sizes: Sequence[int] = DEFAULT_NGRAM_SIZES,
    threshold: float = 0.8,
    top_k: int = 5,
    seed_source: str = "commitpack",
    oneshot_path: Path = DEFAULT_ONESHOT,
) -> List[Dict[str, object]]:
    """Run independent scans for multiple n-gram sizes.

    The scans are deliberately sequential so that only one benchmark inverted
    index is resident in memory at a time. Each size gets an isolated output
    directory, while the root summary index contains all subset/size rows.
    """

    sizes = tuple(dict.fromkeys(ngram_sizes))
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("ngram_sizes must contain positive integers")

    combined_rows: List[Dict[str, object]] = []
    for ngram_size in sizes:
        size_output_dir = output_dir / f"ngram_{ngram_size}"
        rows = run_detection(
            commitpack_path=commitpack_path,
            canitedit_path=canitedit_path,
            codeeditor_dir=codeeditor_dir,
            output_dir=size_output_dir,
            ngram_size=ngram_size,
            threshold=threshold,
            top_k=top_k,
            seed_source=seed_source,
            oneshot_path=oneshot_path,
        )
        for row in rows:
            combined_rows.append({"ngram_size": ngram_size, **row})

    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "ngram_size",
        "subset",
        "total_tasks",
        "contaminated_tasks",
        "contamination_rate",
        "total_code_fields",
        "max_containment",
    ]
    with (output_dir / "summary_index.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(combined_rows)
    return combined_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check CommitPackFT old_contents or one-shot code_before for "
            "token n-gram overlap with CanItEdit and individual "
            "CodeEditorBench subsets."
        ),
        epilog=(
            "Example:\n"
            "  python benchmark/check_ngram_contamination.py "
            "--seed-source oneshot\n"
            "  python benchmark/check_ngram_contamination.py "
            "--seed-source oneshot --ngram-size 10 "
            "--output-dir /tmp/oneshot-contamination"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--seed-source",
        choices=("commitpack", "oneshot"),
        default="commitpack",
        help="seed data to scan (default: commitpack)",
    )
    parser.add_argument("--commitpack", type=Path, default=DEFAULT_COMMITPACK)
    parser.add_argument("--oneshot", type=Path, default=DEFAULT_ONESHOT)
    parser.add_argument("--canitedit", type=Path, default=DEFAULT_CANITEDIT)
    parser.add_argument(
        "--codeeditor-dir", type=Path, default=DEFAULT_CODEEDITOR_DIR
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "result directory (default: benchmark/data/contamination_results "
            "for CommitPackFT, with /oneshot appended for one-shot seeds)"
        ),
    )
    ngram_group = parser.add_mutually_exclusive_group()
    ngram_group.add_argument(
        "--ngram-size",
        type=int,
        help="run one n-gram size and keep the legacy flat output layout",
    )
    ngram_group.add_argument(
        "--ngram-sizes",
        type=int,
        nargs="+",
        help="run multiple sizes (default: 5 10 20)",
    )
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def default_output_dir(seed_source: str) -> Path:
    if seed_source == "commitpack":
        return DEFAULT_OUTPUT_DIR
    if seed_source == "oneshot":
        return DEFAULT_OUTPUT_DIR / "oneshot"
    raise ValueError(f"Unsupported seed source: {seed_source}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or default_output_dir(args.seed_source)
    ngram_sizes = (
        [args.ngram_size]
        if args.ngram_size is not None
        else (args.ngram_sizes or list(DEFAULT_NGRAM_SIZES))
    )
    if any(size <= 0 for size in ngram_sizes):
        raise SystemExit("n-gram sizes must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be between 0 and 1")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    if args.ngram_size is not None:
        rows = [
            {"ngram_size": args.ngram_size, **row}
            for row in run_detection(
                commitpack_path=args.commitpack,
                canitedit_path=args.canitedit,
                codeeditor_dir=args.codeeditor_dir,
                output_dir=output_dir,
                ngram_size=args.ngram_size,
                threshold=args.threshold,
                top_k=args.top_k,
                seed_source=args.seed_source,
                oneshot_path=args.oneshot,
            )
        ]
    else:
        rows = run_multi_detection(
            commitpack_path=args.commitpack,
            canitedit_path=args.canitedit,
            codeeditor_dir=args.codeeditor_dir,
            output_dir=output_dir,
            ngram_sizes=ngram_sizes,
            threshold=args.threshold,
            top_k=args.top_k,
            seed_source=args.seed_source,
            oneshot_path=args.oneshot,
        )
    for row in rows:
        print(
            f"{row['ngram_size']}-gram {row['subset']}: "
            f"{row['contaminated_tasks']}/"
            f"{row['total_tasks']} contaminated "
            f"({float(row['contamination_rate']):.2%})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
