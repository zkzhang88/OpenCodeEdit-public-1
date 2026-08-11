#!/usr/bin/env python3
"""Streamingly check the static quality of OCEData edit pairs."""

from __future__ import annotations

import argparse
import ast
import builtins
from collections import Counter
from dataclasses import dataclass, field
import io
import json
import os
from pathlib import Path
import re
import sys
import symtable
import tempfile
import tokenize
from typing import Dict, Iterable, List, Optional, Sequence, Set, TextIO, Tuple
import warnings

from tqdm import tqdm
import yaml


# Default paths are anchored to the repository, so the script behaves the same
# whether it is launched from the repository root or from generation/.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "OCEData" / "ocedata.jsonl"

# These fragments cover parser/tokenizer messages that strongly indicate a
# truncated snippet rather than an ordinary syntax mistake.
FENCE_RE = re.compile(r"^[ \t]*(?:`{3,}|~{3,})")
INCOMPLETE_MESSAGES = (
    "unexpected eof",
    "eof while scanning",
    "eof in multi-line",
    "unterminated string",
    "was never closed",
    "expected an indented block",
    "incomplete input",
    "unexpected character after line continuation",
)
PYTHON2_BUILTINS = {
    "StandardError",
    "apply",
    "basestring",
    "buffer",
    "cmp",
    "coerce",
    "execfile",
    "file",
    "intern",
    "long",
    "raw_input",
    "reduce",
    "reload",
    "unichr",
    "unicode",
    "xrange",
}
RUNTIME_GLOBALS = {
    "__builtins__",
    "__cached__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
}
BUILTIN_NAMES = set(dir(builtins)) | PYTHON2_BUILTINS | RUNTIME_GLOBALS
STDLIB_MODULES = set(getattr(sys, "stdlib_module_names", ()))
DEFAULT_INSTRUCTION_FIELDS = (
    "instruct_descriptive_purify",
    "instruct_lazy_purify",
)


# ---------------------------------------------------------------------------
# Stable report and analysis data models
# ---------------------------------------------------------------------------

def make_issue(
    code: str,
    side: str,
    field_name: Optional[str],
    message: str,
    *,
    name: Optional[str] = None,
    lineno: Optional[int] = None,
    column: Optional[int] = None,
) -> Dict[str, object]:
    """Build a stable, JSON-serializable issue record."""
    return {
        "code": code,
        "side": side,
        "field": field_name,
        "name": name,
        "line": lineno,
        "column": column,
        "message": message,
    }


@dataclass
class NameUse:
    name: str
    lineno: int
    column: int
    attribute_base: bool = False


@dataclass
class Scope:
    kind: str
    parent: Optional["Scope"] = None
    bindings: Set[str] = field(default_factory=set)
    imports: Dict[str, str] = field(default_factory=dict)
    globals: Set[str] = field(default_factory=set)
    nonlocals: Set[str] = field(default_factory=set)
    loads: List[NameUse] = field(default_factory=list)
    children: List["Scope"] = field(default_factory=list)
    has_star_import: bool = False

    def add_child(self, kind: str) -> "Scope":
        child = Scope(kind=kind, parent=self)
        self.children.append(child)
        return child


@dataclass
class StaticAnalysis:
    dialect: str
    root_scope: Scope
    undefined: Dict[str, NameUse]
    import_aliases: Set[str]
    all_bindings: Set[str]


@dataclass
class ParsedCode:
    dialect: Optional[str]
    tree: Optional[ast.AST]
    normalized_source: Optional[str]
    issues: List[Dict[str, object]]


class ScopeBuilder(ast.NodeVisitor):
    """Collect lexical bindings and name reads without executing code."""

    def __init__(self) -> None:
        self.root = Scope("module")
        self.scope = self.root
        self._attribute_base_nodes: Set[int] = set()

    def build(self, tree: ast.AST) -> Scope:
        self.visit(tree)
        return self.root

    def _visit_in_child(self, child: Scope, nodes: Iterable[ast.AST]) -> None:
        previous = self.scope
        self.scope = child
        try:
            for node in nodes:
                self.visit(node)
        finally:
            self.scope = previous

    def _bind_arguments(self, arguments: ast.arguments, scope: Scope) -> None:
        args = (
            list(arguments.posonlyargs)
            + list(arguments.args)
            + list(arguments.kwonlyargs)
        )
        if arguments.vararg:
            args.append(arguments.vararg)
        if arguments.kwarg:
            args.append(arguments.kwarg)
        scope.bindings.update(arg.arg for arg in args)

    def _visit_function(
        self, node: ast.AST, name: Optional[str], arguments: ast.arguments,
        body: Sequence[ast.AST], decorators: Sequence[ast.AST],
        returns: Optional[ast.AST],
    ) -> None:
        if name:
            self.scope.bindings.add(name)
        for decorator in decorators:
            self.visit(decorator)
        for default in list(arguments.defaults) + [
            item for item in arguments.kw_defaults if item is not None
        ]:
            self.visit(default)
        # Annotations are deliberately ignored: forward and runtime-provided
        # annotation names are not high-confidence undefined references.
        child = self.scope.add_child("function")
        self._bind_arguments(arguments, child)
        self._visit_in_child(child, body)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(
            node, node.name, node.args, node.body, node.decorator_list, node.returns
        )

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(
            node, node.name, node.args, node.body, node.decorator_list, node.returns
        )

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in list(node.args.defaults) + [
            item for item in node.args.kw_defaults if item is not None
        ]:
            self.visit(default)
        child = self.scope.add_child("function")
        self._bind_arguments(node.args, child)
        self._visit_in_child(child, [node.body])

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.bindings.add(node.name)
        for expression in list(node.decorator_list) + list(node.bases):
            self.visit(expression)
        for keyword in node.keywords:
            self.visit(keyword.value)
        child = self.scope.add_child("class")
        self._visit_in_child(child, node.body)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            if node.id not in self.scope.globals and node.id not in self.scope.nonlocals:
                self.scope.bindings.add(node.id)
        elif isinstance(node.ctx, ast.Load):
            self.scope.loads.append(
                NameUse(
                    node.id,
                    getattr(node, "lineno", 0),
                    getattr(node, "col_offset", 0),
                    id(node) in self._attribute_base_nodes,
                )
            )

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Remember the root of expressions such as ``os.path.join``. An
        # unresolved root that names a stdlib module is a strong missing-import
        # signal; the attribute names themselves are intentionally not checked.
        value = node.value
        while isinstance(value, ast.Attribute):
            value = value.value
        if isinstance(value, ast.Name):
            self._attribute_base_nodes.add(id(value))
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            bound = alias.asname or alias.name.split(".", 1)[0]
            self.scope.bindings.add(bound)
            self.scope.imports[bound] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            if alias.name == "*":
                self.scope.has_star_import = True
                continue
            bound = alias.asname or alias.name
            self.scope.bindings.add(bound)
            self.scope.imports[bound] = f"{module}.{alias.name}".strip(".")

    def visit_Global(self, node: ast.Global) -> None:
        self.scope.globals.update(node.names)
        self.scope.bindings.difference_update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.scope.nonlocals.update(node.names)
        self.scope.bindings.difference_update(node.names)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        # ``value += 1`` reads the previous value before storing the result, so
        # the target must be recorded as both a load and a binding.
        if isinstance(node.target, ast.Name):
            self.scope.loads.append(
                NameUse(
                    node.target.id,
                    getattr(node.target, "lineno", 0),
                    getattr(node.target, "col_offset", 0),
                )
            )
            if (
                node.target.id not in self.scope.globals
                and node.target.id not in self.scope.nonlocals
            ):
                self.scope.bindings.add(node.target.id)
        else:
            self.visit(node.target)
        self.visit(node.value)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type:
            self.visit(node.type)
        if isinstance(node.name, str):
            self.scope.bindings.add(node.name)
        for statement in node.body:
            self.visit(statement)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.pattern:
            self.visit(node.pattern)
        if node.name:
            self.scope.bindings.add(node.name)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name:
            self.scope.bindings.add(node.name)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        for key in node.keys:
            self.visit(key)
        for pattern in node.patterns:
            self.visit(pattern)
        if node.rest:
            self.scope.bindings.add(node.rest)

    def _visit_comprehension(
        self,
        generators: Sequence[ast.comprehension],
        results: Sequence[ast.AST],
    ) -> None:
        if not generators:
            for result in results:
                self.visit(result)
            return
        # Python evaluates the first iterable in the surrounding scope, then
        # creates the comprehension scope for targets, filters, and results.
        self.visit(generators[0].iter)
        child = self.scope.add_child("comprehension")
        previous = self.scope
        self.scope = child
        try:
            self.visit(generators[0].target)
            for condition in generators[0].ifs:
                self.visit(condition)
            for generator in generators[1:]:
                self.visit(generator.iter)
                self.visit(generator.target)
                for condition in generator.ifs:
                    self.visit(condition)
            for result in results:
                self.visit(result)
        finally:
            self.scope = previous

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node.generators, [node.key, node.value])


def _module_scope(scope: Scope) -> Scope:
    while scope.parent is not None:
        scope = scope.parent
    return scope


def _parent_visible_from(scope: Scope) -> Optional[Scope]:
    parent = scope.parent
    # Function bodies, lambdas, and comprehensions do not close over class
    # namespaces for bare-name lookup.
    if scope.kind in {"function", "comprehension"} and parent and parent.kind == "class":
        return parent.parent
    if scope.kind == "class" and parent and parent.kind == "class":
        return parent.parent
    return parent


def _is_resolved(name: str, scope: Scope) -> bool:
    if name in BUILTIN_NAMES:
        return True
    if name in scope.globals:
        module = _module_scope(scope)
        return name in module.bindings or name in module.imports or module.has_star_import
    if name in scope.nonlocals:
        parent = scope.parent
        while parent is not None and parent.kind != "module":
            if parent.kind != "class" and name in parent.bindings:
                return True
            parent = parent.parent
        return False
    current: Optional[Scope] = scope
    while current is not None:
        if name in current.bindings or name in current.imports:
            return True
        # A star import can provide arbitrary names. Suppressing findings in
        # this case favors precision over potentially noisy guesses.
        if current.has_star_import:
            return True
        current = _parent_visible_from(current)
    return False


def _walk_scopes(scope: Scope) -> Iterable[Scope]:
    yield scope
    for child in scope.children:
        yield from _walk_scopes(child)


def analyze_tree(tree: ast.AST, dialect: str) -> StaticAnalysis:
    builder = ScopeBuilder()
    root = builder.build(tree)
    undefined: Dict[str, NameUse] = {}
    import_aliases: Set[str] = set()
    all_bindings: Set[str] = set()
    for scope in _walk_scopes(root):
        import_aliases.update(scope.imports)
        all_bindings.update(scope.bindings)
        for use in scope.loads:
            if not _is_resolved(use.name, scope):
                previous = undefined.get(use.name)
                # Keep one diagnostic per name, preferring an attribute-root
                # occurrence because it can support missing-import detection.
                if previous is None or (use.attribute_base and not previous.attribute_base):
                    undefined[use.name] = use
    return StaticAnalysis(
        dialect=dialect,
        root_scope=root,
        undefined=undefined,
        import_aliases=import_aliases,
        all_bindings=all_bindings,
    )


class Python2Converter:
    """Lazily load deprecated stdlib machinery only when fallback is needed."""

    def __init__(self) -> None:
        self._tool = None
        self._unavailable_reason: Optional[str] = None

    def convert(self, source: str, filename: str) -> str:
        if self._unavailable_reason:
            raise RuntimeError(self._unavailable_reason)
        if self._tool is None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    from lib2to3.refactor import (  # type: ignore[import-not-found]
                        RefactoringTool,
                        get_fixers_from_package,
                    )

                    self._tool = RefactoringTool(
                        get_fixers_from_package("lib2to3.fixes")
                    )
            except Exception as exc:  # pragma: no cover - Python 3.13+
                self._unavailable_reason = f"Python 2 parser unavailable: {exc}"
                raise RuntimeError(self._unavailable_reason) from exc
        parse_source = source if source.endswith("\n") else source + "\n"
        return str(self._tool.refactor_string(parse_source, filename))


PYTHON2_CONVERTER = Python2Converter()


# ---------------------------------------------------------------------------
# Syntax, truncation, and dialect handling
# ---------------------------------------------------------------------------

def _syntax_location(exc: BaseException) -> Tuple[Optional[int], Optional[int]]:
    return (
        getattr(exc, "lineno", None),
        getattr(exc, "offset", None),
    )


def _token_incomplete_error(source: str) -> Optional[Tuple[str, int, int]]:
    try:
        list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError) as exc:
        message = str(exc.args[0] if exc.args else exc)
        location = exc.args[1] if len(exc.args) > 1 else (None, None)
        if any(fragment in message.lower() for fragment in INCOMPLETE_MESSAGES):
            return message, location[0], location[1]
    return None


def contains_markdown_fence(source: str) -> bool:
    """Detect an outer Markdown fence without flagging docstring underlines."""
    nonempty_lines = [line for line in source.splitlines() if line.strip()]
    if not nonempty_lines:
        return False
    return bool(
        FENCE_RE.match(nonempty_lines[0]) or FENCE_RE.match(nonempty_lines[-1])
    )


def _layout_insensitive_tokens(source: str) -> Tuple[Tuple[int, str], ...]:
    """Return tokens with blank lines and formatting whitespace normalized."""
    normalized = []
    for token_info in tokenize.generate_tokens(io.StringIO(source).readline):
        token_type, token_text = token_info.type, token_info.string
        if token_type in {tokenize.NL, tokenize.ENDMARKER}:
            continue
        if token_type in {tokenize.INDENT, tokenize.NEWLINE}:
            # Keep block/statement boundaries while ignoring their exact
            # whitespace representation.
            token_text = ""
        normalized.append((token_type, token_text))
    return tuple(normalized)


def code_is_equivalent(pre_code: str, post_code: str) -> bool:
    """Compare code while ignoring blank lines and formatting whitespace."""
    try:
        return _layout_insensitive_tokens(pre_code) == _layout_insensitive_tokens(
            post_code
        )
    except (tokenize.TokenError, IndentationError):
        # Tokenization can fail for already malformed snippets. A conservative
        # text fallback still ignores blank lines and surrounding whitespace.
        def normalize_lines(source: str) -> Tuple[str, ...]:
            return tuple(
                line.strip() for line in source.splitlines() if line.strip()
            )

        return normalize_lines(pre_code) == normalize_lines(post_code)


def _is_incomplete_syntax(source: str, exc: BaseException) -> bool:
    message = str(exc).lower()
    if any(fragment in message for fragment in INCOMPLETE_MESSAGES):
        return True
    lineno = getattr(exc, "lineno", None)
    last_line = max(1, len(source.splitlines()))
    if lineno is not None and lineno >= last_line:
        return any(
            fragment in message
            for fragment in ("expected 'except'", "expected 'finally'", "invalid syntax")
        )
    return False


def parse_code(source: str, side: str, field_name: str) -> ParsedCode:
    py3_error: Optional[BaseException] = None
    # First validate with the running Python 3 interpreter. Building a symbol
    # table and bytecode catches semantic syntax errors that ast.parse alone
    # may accept, while still never executing the sample.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source, filename=f"<{side}>", mode="exec")
            symtable.symtable(source, f"<{side}>", "exec")
            compile(source, f"<{side}>", "exec")
        return ParsedCode("python3", tree, source, [])
    except (SyntaxError, ValueError, OverflowError) as exc:
        py3_error = exc

    # Historical snippets may contain valid Python 2 syntax. Convert only for
    # analysis; the original JSONL text is always preserved in output.
    try:
        converted = PYTHON2_CONVERTER.convert(source, f"<{side}>")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(converted, filename=f"<{side}:python2>", mode="exec")
            symtable.symtable(converted, f"<{side}:python2>", "exec")
            compile(converted, f"<{side}:python2>", "exec")
        return ParsedCode("python2", tree, converted, [])
    except Exception as py2_error:
        assert py3_error is not None
        lineno, offset = _syntax_location(py3_error)
        lower_message = str(py3_error).lower()
        if "nonlocal" in lower_message or "global declaration" in lower_message:
            code = "unresolvable_reference"
        else:
            code = "syntax_error"
        issues = [
            make_issue(
                code,
                side,
                field_name,
                f"Python 3: {py3_error}; Python 2 fallback: {py2_error}",
                lineno=lineno,
                column=offset,
            )
        ]
        token_error = _token_incomplete_error(source)
        # Syntax errors and truncation are reported separately so consumers can
        # distinguish malformed code from likely cut-off model output.
        if _is_incomplete_syntax(source, py3_error) or token_error:
            token_message = token_error[0] if token_error else str(py3_error)
            token_line = token_error[1] if token_error else lineno
            token_column = token_error[2] if token_error else offset
            issues.append(
                make_issue(
                    "incomplete_structure",
                    side,
                    field_name,
                    token_message,
                    lineno=token_line,
                    column=token_column,
                )
            )
        return ParsedCode(None, None, None, issues)


def _undefined_issue_code(
    name: str,
    use: NameUse,
    current: StaticAnalysis,
    other: Optional[StaticAnalysis],
    side: str,
) -> str:
    # Cross-checking the other edit side makes removed imports/definitions much
    # more reliable than guessing solely from an isolated code fragment.
    if (
        (name in STDLIB_MODULES and use.attribute_base)
        or (
            other is not None
            and name in other.import_aliases
            and name not in current.import_aliases
        )
    ):
        return "missing_import"
    # A binding can only become unresolvable in the forward edit direction.
    # When checking pre-edit, a definition added by post-edit does not make the
    # reference resolvable in the pre-edit code; it is simply undefined there.
    if side == "post" and other is not None and name in other.all_bindings:
        return "unresolvable_reference"
    return "undefined_name"


def reference_issues(
    analysis: StaticAnalysis,
    side: str,
    field_name: str,
    other: Optional[StaticAnalysis],
) -> List[Dict[str, object]]:
    issues = []
    for name, use in sorted(
        analysis.undefined.items(), key=lambda item: (item[1].lineno, item[0])
    ):
        code = _undefined_issue_code(name, use, analysis, other, side)
        if code == "missing_import":
            message = f"Name '{name}' is used like a module but is not imported"
        elif code == "unresolvable_reference":
            message = f"Reference to name '{name}' cannot be resolved at this location"
        else:
            message = f"Name '{name}' has no visible binding"
        issues.append(
            make_issue(
                code,
                side,
                field_name,
                message,
                name=name,
                lineno=use.lineno,
                column=use.column,
            )
        )
    return issues


def inspect_pair(
    record: Dict[str, object],
    pre_field: str,
    post_field: str,
    instruction_field: Optional[str] = None,
    *,
    instruction_fields: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, object]], Counter]:
    issues: List[Dict[str, object]] = []
    parse_counts: Counter = Counter()
    values: Dict[str, Optional[str]] = {"pre": None, "post": None}
    analyses: Dict[str, StaticAnalysis] = {}
    parsed_codes: Dict[str, ParsedCode] = {}
    pre_incomplete_issues: List[Dict[str, object]] = []
    post_parse_issues: List[Dict[str, object]] = []

    configured_instruction_fields = _resolve_instruction_fields(
        instruction_field, instruction_fields
    )
    for configured_field in configured_instruction_fields:
        if configured_field not in record:
            issues.append(
                make_issue(
                    "missing_field",
                    "instruction",
                    configured_field,
                    f"Required field '{configured_field}' is missing",
                )
            )
            continue
        instruction = record[configured_field]
        if not isinstance(instruction, str):
            issues.append(
                make_issue(
                    "invalid_field_type",
                    "instruction",
                    configured_field,
                    f"Expected a string, got {type(instruction).__name__}",
                )
            )
            continue
        if not instruction.strip():
            issues.append(
                make_issue(
                    "empty_instruction",
                    "instruction",
                    configured_field,
                    "Edit instruction is empty",
                )
            )

    # Parse both sides so pre-edit can serve as a silent comparison baseline,
    # but defer pre-edit truncation findings until post-edit has been parsed.
    # A valid post-edit proves that the edit repaired the incomplete input.
    for side, field_name in (("pre", pre_field), ("post", post_field)):
        if field_name not in record:
            issues.append(
                make_issue(
                    "missing_field",
                    side,
                    field_name,
                    f"Required field '{field_name}' is missing",
                )
            )
            continue
        value = record[field_name]
        if not isinstance(value, str):
            issues.append(
                make_issue(
                    "invalid_field_type",
                    side,
                    field_name,
                    f"Expected a string, got {type(value).__name__}",
                )
            )
            continue
        values[side] = value
        if not value.strip():
            issues.append(
                make_issue("empty_code", side, field_name, "Code output is empty")
            )
            continue
        if contains_markdown_fence(value):
            issues.append(
                make_issue(
                    "markdown_fence",
                    side,
                    field_name,
                    "Code contains a Markdown fence",
                )
            )
        parsed_code = parse_code(value, side, field_name)
        parsed_codes[side] = parsed_code
        if side == "pre":
            pre_incomplete_issues.extend(
                issue
                for issue in parsed_code.issues
                if issue["code"] == "incomplete_structure"
            )
        else:
            post_parse_issues.extend(parsed_code.issues)
        if parsed_code.dialect:
            parse_counts[parsed_code.dialect] += 1
        else:
            parse_counts["unparseable"] += 1
        if parsed_code.tree is not None and parsed_code.dialect is not None:
            analyses[side] = analyze_tree(parsed_code.tree, parsed_code.dialect)

    post_parsed = parsed_codes.get("post")
    if post_parsed is None or post_parsed.dialect is None:
        issues.extend(pre_incomplete_issues)
    issues.extend(post_parse_issues)

    if values["pre"] is not None and values["post"] is not None:
        if code_is_equivalent(values["pre"], values["post"]):
            issues.append(
                make_issue(
                    "identical_code",
                    "pair",
                    None,
                    "Pre-edit and post-edit code are equivalent after "
                    "ignoring blank lines and layout whitespace",
                )
            )

    pre_analysis = analyses.get("pre")
    post_analysis = analyses.get("post")
    if post_analysis:
        post_reference_issues = reference_issues(
            post_analysis, "post", post_field, pre_analysis
        )
        issues.extend(post_reference_issues)
        # Only claim that post-edit introduced an issue when a valid pre-edit
        # analysis exists. Otherwise there is no trustworthy comparison base.
        if pre_analysis:
            pre_undefined = set(pre_analysis.undefined)
            for issue in post_reference_issues:
                name = issue.get("name")
                if isinstance(name, str) and name not in pre_undefined:
                    new_issue = dict(issue)
                    new_issue["code"] = f"new_{issue['code']}"
                    new_issue["message"] = f"Post-edit introduced: {issue['message']}"
                    issues.append(new_issue)

    return issues, parse_counts


def _resolve_instruction_fields(
    instruction_field: Optional[str],
    instruction_fields: Optional[Sequence[str]],
) -> Tuple[str, ...]:
    """Resolve the legacy singular option and the multi-field configuration."""
    if instruction_field is not None and instruction_fields is not None:
        raise ValueError(
            "instruction_field and instruction_fields cannot both be specified"
        )
    if instruction_fields is not None:
        configured = (
            (instruction_fields,)
            if isinstance(instruction_fields, str)
            else instruction_fields
        )
    elif instruction_field is not None:
        configured = (instruction_field,)
    else:
        configured = DEFAULT_INSTRUCTION_FIELDS
    if not configured:
        raise ValueError("At least one instruction field must be configured")

    resolved: List[str] = []
    for field_name in configured:
        if not isinstance(field_name, str) or not field_name:
            raise ValueError("Instruction field names must be non-empty strings")
        if field_name not in resolved:
            resolved.append(field_name)
    return tuple(resolved)


def _validate_paths(
    input_file: Path,
    report_file: Path,
    filtered_file: Path,
    summary_file: Optional[Path] = None,
) -> None:
    paths = [input_file, report_file, filtered_file]
    if summary_file is not None:
        paths.append(summary_file)
    resolved = [path.expanduser().resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Input and output paths must all be different")
    if not resolved[0].is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")


def _temporary_output(path: Path) -> Tuple[TextIO, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        delete=False,
    )
    return handle, Path(handle.name)


def check_jsonl(
    input_file: Path,
    report_file: Path,
    filtered_file: Path,
    pre_field: str = "code_before_purify",
    post_field: str = "code_after_purify",
    show_progress: bool = True,
    summary_file: Optional[Path] = None,
    instruction_field: Optional[str] = None,
    instruction_fields: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Check JSONL and atomically write issue, filtered, and summary outputs."""
    _validate_paths(input_file, report_file, filtered_file, summary_file)
    configured_instruction_fields = _resolve_instruction_fields(
        instruction_field, instruction_fields
    )
    # Count records without loading them so tqdm can display a percentage and ETA.
    record_count = None
    if show_progress:
        with input_file.open("r", encoding="utf-8") as count_handle:
            record_count = sum(1 for _ in count_handle)

    issue_counts: Counter = Counter()
    parse_counts: Counter = Counter()
    total = passed = failed = 0
    report_handle, report_tmp = _temporary_output(report_file)
    filtered_handle, filtered_tmp = _temporary_output(filtered_file)
    progress = None
    try:
        # Process one line at a time to avoid loading the roughly 177 MB source
        # dataset or the generated outputs into memory.
        with input_file.open("r", encoding="utf-8") as input_handle:
            progress = tqdm(
                enumerate(input_handle, 1),
                total=record_count,
                desc="Checking OCEData",
                unit="record",
                dynamic_ncols=True,
                disable=not show_progress,
            )
            for line_number, raw_line in progress:
                total += 1
                issues: List[Dict[str, object]]
                record: Optional[Dict[str, object]] = None
                try:
                    loaded = json.loads(raw_line)
                    if not isinstance(loaded, dict):
                        raise TypeError(
                            f"Expected a JSON object, got {type(loaded).__name__}"
                        )
                    record = loaded
                    issues, row_parse_counts = inspect_pair(
                        record,
                        pre_field,
                        post_field,
                        instruction_fields=configured_instruction_fields,
                    )
                    parse_counts.update(row_parse_counts)
                except (json.JSONDecodeError, TypeError) as exc:
                    issues = [
                        make_issue(
                            "invalid_json",
                            "record",
                            None,
                            str(exc),
                            lineno=line_number,
                        )
                    ]

                if issues:
                    failed += 1
                    issue_counts.update(str(issue["code"]) for issue in issues)
                    report_record = {
                        "line_number": line_number,
                        "commit": record.get("commit") if record else None,
                        "instr_type": record.get("instr_type") if record else None,
                        "issues": issues,
                    }
                    if record is None:
                        report_record["raw_excerpt"] = raw_line[:500].rstrip("\n")
                    report_handle.write(
                        json.dumps(report_record, ensure_ascii=False) + "\n"
                    )
                else:
                    passed += 1
                    filtered_handle.write(
                        raw_line if raw_line.endswith("\n") else raw_line + "\n"
                    )

                if show_progress:
                    progress.set_postfix(
                        passed=f"{passed:,}", failed=f"{failed:,}", refresh=False
                    )

            progress.close()

        report_handle.flush()
        filtered_handle.flush()
        os.fsync(report_handle.fileno())
        os.fsync(filtered_handle.fileno())
        report_handle.close()
        filtered_handle.close()
        # Replace completed outputs only after both temporary files have been
        # fully flushed, preventing interrupted runs from leaving partial JSONL.
        os.replace(report_tmp, report_file)
        os.replace(filtered_tmp, filtered_file)
    except Exception:
        if progress is not None:
            progress.close()
        report_handle.close()
        filtered_handle.close()
        for temporary in (report_tmp, filtered_tmp):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise

    summary: Dict[str, object] = {
        "python_runtime": sys.version.split()[0],
        "total": total,
        "passed": passed,
        "failed": failed,
        "parse_counts": dict(sorted(parse_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "report_file": str(report_file),
        "filtered_file": str(filtered_file),
    }
    if summary_file is not None:
        summary["summary_file"] = str(summary_file)
        write_summary_yaml(summary, summary_file)
    return summary


def write_summary_yaml(summary: Dict[str, object], summary_file: Path) -> None:
    """Atomically write the terminal summary data as UTF-8 YAML."""
    summary_handle, summary_tmp = _temporary_output(summary_file)
    try:
        yaml.safe_dump(
            summary,
            summary_handle,
            allow_unicode=True,
            sort_keys=False,
        )
        summary_handle.flush()
        os.fsync(summary_handle.fileno())
        summary_handle.close()
        os.replace(summary_tmp, summary_file)
    except Exception:
        summary_handle.close()
        try:
            summary_tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def print_summary(summary: Dict[str, object]) -> None:
    print(f"Python runtime: {summary['python_runtime']}")
    print(
        f"Records: {summary['total']} total, "
        f"{summary['passed']} passed, {summary['failed']} failed"
    )
    print("Parser results:")
    for name, count in dict(summary["parse_counts"]).items():
        print(f"  {name}: {count}")
    print("Issues:")
    for name, count in dict(summary["issue_counts"]).items():
        print(f"  {name}: {count}")
    print(f"Issue report: {summary['report_file']}")
    print(f"Filtered data: {summary['filtered_file']}")
    if "summary_file" in summary:
        print(f"Summary YAML: {summary['summary_file']}")


def default_output_path(input_file: Path, suffix: str) -> Path:
    """Derive a sibling JSONL output path from the input file name."""
    extension = input_file.suffix or ".jsonl"
    return input_file.with_name(f"{input_file.stem}{suffix}{extension}")


def default_summary_path(input_file: Path) -> Path:
    """Derive the sibling YAML summary path from the input file name."""
    return input_file.with_name(f"{input_file.stem}_quality_summary.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Statically check edit instructions and pre-edit/post-edit Python "
            "in OCEData."
        )
    )
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--report-file",
        type=Path,
        default=None,
        help="Issue report path (default: <input_stem>_quality_issues.jsonl).",
    )
    parser.add_argument(
        "--filtered-file",
        type=Path,
        default=None,
        help="Filtered output path (default: <input_stem>_quality_filtered.jsonl).",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="YAML summary path (default: <input_stem>_quality_summary.yaml).",
    )
    parser.add_argument("--pre-field", default="code_before_purify")
    parser.add_argument("--post-field", default="code_after_purify")
    parser.add_argument(
        "--instruction-fields",
        "--instruction-field",
        dest="instruction_fields",
        nargs="+",
        default=DEFAULT_INSTRUCTION_FIELDS,
        metavar="FIELD",
        help=(
            "Instruction fields that must contain non-empty strings "
            "(defaults: instruct_descriptive_purify and "
            "instruct_lazy_purify)."
        ),
    )
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Return exit status 1 when at least one record has issues.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report_file = args.report_file or default_output_path(
        args.input_file, "_quality_issues"
    )
    filtered_file = args.filtered_file or default_output_path(
        args.input_file, "_quality_filtered"
    )
    summary_file = args.summary_file or default_summary_path(args.input_file)
    try:
        summary = check_jsonl(
            input_file=args.input_file,
            report_file=report_file,
            filtered_file=filtered_file,
            summary_file=summary_file,
            pre_field=args.pre_field,
            post_field=args.post_field,
            instruction_fields=args.instruction_fields,
            show_progress=not args.no_progress,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print_summary(summary)
    return 1 if args.fail_on_issues and summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
