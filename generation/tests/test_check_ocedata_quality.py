import json
from pathlib import Path
import tempfile
import unittest

from generation.check_ocedata_quality import (
    check_jsonl,
    code_is_equivalent,
    contains_markdown_fence,
    inspect_pair,
    main,
)


PRE = "code_before_purify"
POST = "code_after_purify"


def issue_codes(record):
    issues, _ = inspect_pair(record, PRE, POST)
    return [issue["code"] for issue in issues]


class StaticAnalysisTests(unittest.TestCase):
    def test_valid_scopes_imports_builtins_and_comprehensions(self):
        code = """import math
value = 2

def outer(arg):
    local = [math.sqrt(item) for item in range(arg)]
    def inner():
        return local
    return inner()
"""
        self.assertEqual(issue_codes({PRE: code, POST: code + "\nresult = value"}), [])

    def test_existing_external_name_is_not_new(self):
        before = "result = framework_value()\n"
        after = "result = framework_value()\nprint(result)\n"
        issues, _ = inspect_pair({PRE: before, POST: after}, PRE, POST)
        self.assertIn("undefined_name", [item["code"] for item in issues])
        self.assertNotIn("new_undefined_name", [item["code"] for item in issues])

    def test_new_undefined_name(self):
        issues, _ = inspect_pair(
            {PRE: "value = 1\n", POST: "value = missing_name()\n"}, PRE, POST
        )
        codes = [item["code"] for item in issues]
        self.assertIn("undefined_name", codes)
        self.assertIn("new_undefined_name", codes)

    def test_removed_definition_becomes_unresolvable(self):
        before = "def helper():\n    return 1\nresult = helper()\n"
        after = "result = helper()\n"
        codes = issue_codes({PRE: before, POST: after})
        self.assertIn("unresolvable_reference", codes)
        self.assertIn("new_unresolvable_reference", codes)

    def test_removed_import_and_stdlib_module_are_missing_imports(self):
        before = "import json\nresult = json.dumps({})\n"
        after = "result = json.dumps({})\nother = os.path.join('a', 'b')\n"
        codes = issue_codes({PRE: before, POST: after})
        self.assertGreaterEqual(codes.count("missing_import"), 2)
        self.assertGreaterEqual(codes.count("new_missing_import"), 2)

    def test_global_nonlocal_and_lambda(self):
        before = """state = 0
def outer(value):
    current = value
    def inner():
        nonlocal current
        global state
        current += 1
        state += 1
        return current
    return lambda item: inner() + item
"""
        after = before + "answer = outer(1)(2)\n"
        self.assertEqual(issue_codes({PRE: before, POST: after}), [])

    def test_invalid_nonlocal_is_unresolvable(self):
        code = "def broken():\n    nonlocal missing\n    return missing\n"
        self.assertIn(
            "unresolvable_reference", issue_codes({PRE: "x = 1\n", POST: code})
        )


class FormatAndParsingTests(unittest.TestCase):
    def test_python2_fallback(self):
        issues, counts = inspect_pair(
            {PRE: "print 'before'\n", POST: "print 'after'\n"}, PRE, POST
        )
        self.assertEqual(issues, [])
        self.assertEqual(counts["python2"], 2)

    def test_empty_fence_identical_and_incomplete(self):
        issues, _ = inspect_pair({PRE: "", POST: ""}, PRE, POST)
        codes = [item["code"] for item in issues]
        self.assertEqual(codes.count("empty_code"), 2)
        self.assertIn("identical_code", codes)

        fenced = "```python\nvalue = 1\n```\n"
        self.assertIn("markdown_fence", issue_codes({PRE: "x = 1\n", POST: fenced}))

        incomplete = "def broken():\n"
        codes = issue_codes({PRE: "x = 1\n", POST: incomplete})
        self.assertIn("syntax_error", codes)
        self.assertIn("incomplete_structure", codes)

    def test_identical_code_ignores_layout_but_not_content(self):
        before = "if ready:\n    value = call('a  b')\n\n"
        after = "if  ready:\n  value=call('a  b')\n"
        self.assertTrue(code_is_equivalent(before, after))
        self.assertIn("identical_code", issue_codes({PRE: before, POST: after}))

        changed_string = "if ready:\n    value = call('a b')\n"
        self.assertFalse(code_is_equivalent(before, changed_string))
        self.assertNotIn(
            "identical_code", issue_codes({PRE: before, POST: changed_string})
        )

    def test_rst_underlines_are_not_markdown_fences(self):
        source = '"""\\ntitle\\n~~~~~~~~~~~~\\n"""\nvalue = 1\n'
        self.assertFalse(contains_markdown_fence(source))
        self.assertTrue(contains_markdown_fence("~~~python\nvalue = 1\n~~~\n"))

    def test_missing_and_non_string_fields(self):
        codes = issue_codes({PRE: 3})
        self.assertIn("invalid_field_type", codes)
        self.assertIn("missing_field", codes)


class JsonlIntegrationTests(unittest.TestCase):
    def test_outputs_counts_and_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            report = root / "issues.jsonl"
            filtered = root / "filtered.jsonl"
            valid = {PRE: "x = 1\n", POST: "x = 2\n", "commit": "good"}
            invalid = {PRE: "x = 1\n", POST: "x = missing\n", "commit": "bad"}
            source.write_text(
                json.dumps(valid) + "\n"
                + "{not json}\n"
                + json.dumps(invalid) + "\n",
                encoding="utf-8",
            )

            summary = check_jsonl(
                source, report, filtered, progress_every=0
            )
            self.assertEqual(summary["total"], 3)
            self.assertEqual(summary["passed"], 1)
            self.assertEqual(summary["failed"], 2)
            self.assertEqual(len(report.read_text(encoding="utf-8").splitlines()), 2)
            self.assertEqual(
                json.loads(filtered.read_text(encoding="utf-8"))["commit"], "good"
            )
            self.assertEqual(
                main(
                    [
                        "--input-file", str(source),
                        "--report-file", str(root / "issues-2.jsonl"),
                        "--filtered-file", str(root / "filtered-2.jsonl"),
                        "--fail-on-issues",
                    ]
                ),
                1,
            )

    def test_rejects_overlapping_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.jsonl"
            source.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                check_jsonl(source, source, Path(directory) / "filtered.jsonl")


if __name__ == "__main__":
    unittest.main()
