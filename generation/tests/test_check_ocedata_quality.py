import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

from generation.check_ocedata_quality import (
    check_jsonl,
    code_is_equivalent,
    contains_markdown_fence,
    inspect_pair,
    main,
)


PRE = "code_before_purify"
POST = "code_after_purify"
INSTRUCTION = "instruct_purify"
VALID_INSTRUCTION = "Update the code."


def with_instruction(record):
    return {INSTRUCTION: VALID_INSTRUCTION, **record}


def issue_codes(record):
    issues, _ = inspect_pair(with_instruction(record), PRE, POST)
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
        issues, _ = inspect_pair(
            with_instruction({PRE: before, POST: after}), PRE, POST
        )
        self.assertIn("undefined_name", [item["code"] for item in issues])
        self.assertNotIn("new_undefined_name", [item["code"] for item in issues])

    def test_new_undefined_name(self):
        issues, _ = inspect_pair(
            with_instruction(
                {PRE: "value = 1\n", POST: "value = missing_name()\n"}
            ),
            PRE,
            POST,
        )
        codes = [item["code"] for item in issues]
        self.assertIn("undefined_name", codes)
        self.assertIn("new_undefined_name", codes)

    def test_removed_definition_becomes_unresolvable(self):
        before = "def helper():\n    return 1\nresult = helper()\n"
        after = "result = helper()\n"
        issues, _ = inspect_pair(
            with_instruction({PRE: before, POST: after}), PRE, POST
        )
        codes = [issue["code"] for issue in issues]
        self.assertIn("unresolvable_reference", codes)
        self.assertIn("new_unresolvable_reference", codes)
        messages = [
            issue["message"]
            for issue in issues
            if issue["code"] in {
                "unresolvable_reference",
                "new_unresolvable_reference",
            }
        ]
        self.assertEqual(
            messages,
            [
                "Reference to name 'helper' cannot be resolved at this location",
                "Post-edit introduced: Reference to name 'helper' cannot be "
                "resolved at this location",
            ],
        )
        self.assertTrue(all("previously bound" not in message for message in messages))

    def test_definition_added_by_post_does_not_reverse_pre_issue_direction(self):
        before = "result = parse_article(url)\n"
        after = "def parse_article(url):\n    return url\nresult = parse_article(url)\n"
        issues, _ = inspect_pair(
            with_instruction({PRE: before, POST: after}), PRE, POST
        )
        pre_name_issues = [
            issue
            for issue in issues
            if issue["side"] == "pre" and issue.get("name") == "parse_article"
        ]
        self.assertEqual(pre_name_issues, [])

    def test_pre_static_and_non_truncation_syntax_issues_are_silent(self):
        cases = [
            "result = missing_name()\n",
            "result = os.path.join('a', 'b')\n",
            "def broken():\n    nonlocal missing\n    return missing\n",
            "break\n",
        ]
        for before in cases:
            with self.subTest(before=before):
                issues, _ = inspect_pair(
                    with_instruction({PRE: before, POST: "value = 1\n"}),
                    PRE,
                    POST,
                )
                self.assertEqual(issues, [])

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
            with_instruction({PRE: "print 'before'\n", POST: "print 'after'\n"}),
            PRE,
            POST,
        )
        self.assertEqual(issues, [])
        self.assertEqual(counts["python2"], 2)

    def test_empty_fence_identical_and_incomplete(self):
        issues, _ = inspect_pair(with_instruction({PRE: "", POST: ""}), PRE, POST)
        codes = [item["code"] for item in issues]
        self.assertEqual(codes.count("empty_code"), 2)
        self.assertIn("identical_code", codes)

        fenced = "```python\nvalue = 1\n```\n"
        self.assertIn("markdown_fence", issue_codes({PRE: "x = 1\n", POST: fenced}))

        incomplete = "def broken():\n"
        codes = issue_codes({PRE: "x = 1\n", POST: incomplete})
        self.assertIn("syntax_error", codes)
        self.assertIn("incomplete_structure", codes)

    def test_pre_reports_empty_and_fence_but_allows_repaired_incomplete_code(self):
        empty_codes = issue_codes({PRE: "", POST: "value = 1\n"})
        self.assertEqual(empty_codes, ["empty_code"])

        fenced = "```python\nvalue = 1\n```\n"
        fenced_codes = issue_codes({PRE: fenced, POST: "value = 1\n"})
        self.assertIn("markdown_fence", fenced_codes)
        self.assertNotIn("syntax_error", fenced_codes)

        repaired_cases = [
            (
                "def receiver(func):\n    return func\n\n@receiver\n",
                "def receiver(func):\n    return func\n\n@receiver\n"
                "def handler():\n    pass\n",
            ),
            (
                "settings = object()\nsetattr(settings, 'enabled',\n",
                "settings = object()\nsetattr(settings, 'enabled', True)\n",
            ),
            (
                "def setup(**kwargs):\n    pass\nsetup(name='example',\n",
                "def setup(**kwargs):\n    pass\nsetup(name='example')\n",
            ),
        ]
        for before, after in repaired_cases:
            with self.subTest(before=before):
                incomplete_codes = issue_codes({PRE: before, POST: after})
                self.assertNotIn("incomplete_structure", incomplete_codes)
                self.assertNotIn("syntax_error", incomplete_codes)

    def test_pre_incomplete_structure_remains_when_post_is_unparseable(self):
        issues, _ = inspect_pair(
            with_instruction({PRE: "def broken():\n", POST: "if ready:\n"}),
            PRE,
            POST,
        )
        pre_codes = [
            issue["code"] for issue in issues if issue["side"] == "pre"
        ]
        self.assertIn("incomplete_structure", pre_codes)

    def test_unparseable_pre_does_not_make_post_issue_new(self):
        issues, _ = inspect_pair(
            with_instruction(
                {PRE: "def broken():\n", POST: "result = missing_name()\n"}
            ),
            PRE,
            POST,
        )
        codes = [issue["code"] for issue in issues]
        pre_codes = [
            issue["code"] for issue in issues if issue["side"] == "pre"
        ]
        self.assertIn("undefined_name", codes)
        self.assertNotIn("incomplete_structure", pre_codes)
        self.assertNotIn("new_undefined_name", codes)

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

    def test_instruction_must_be_present_nonempty_string(self):
        code_fields = {PRE: "value = 1\n", POST: "value = 2\n"}
        cases = [
            ({}, "missing_field"),
            ({INSTRUCTION: 3}, "invalid_field_type"),
            ({INSTRUCTION: ""}, "empty_instruction"),
            ({INSTRUCTION: " \n\t"}, "empty_instruction"),
        ]
        for instruction_fields, expected_code in cases:
            with self.subTest(instruction_fields=instruction_fields):
                issues, _ = inspect_pair(
                    {**code_fields, **instruction_fields}, PRE, POST
                )
                instruction_issues = [
                    issue for issue in issues if issue["side"] == "instruction"
                ]
                self.assertEqual(
                    [issue["code"] for issue in instruction_issues],
                    [expected_code],
                )
                self.assertEqual(instruction_issues[0]["field"], INSTRUCTION)

        issues, _ = inspect_pair(with_instruction(code_fields), PRE, POST)
        self.assertEqual(issues, [])


class JsonlIntegrationTests(unittest.TestCase):
    def test_outputs_counts_and_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            report = root / "issues.jsonl"
            filtered = root / "filtered.jsonl"
            summary_file = root / "summary.yaml"
            valid = with_instruction(
                {PRE: "x = 1\n", POST: "x = 2\n", "commit": "good"}
            )
            invalid = with_instruction(
                {PRE: "x = 1\n", POST: "x = missing\n", "commit": "bad"}
            )
            source.write_text(
                json.dumps(valid) + "\n"
                + "{not json}\n"
                + json.dumps(invalid) + "\n",
                encoding="utf-8",
            )

            summary = check_jsonl(
                source,
                report,
                filtered,
                summary_file=summary_file,
                show_progress=False,
            )
            self.assertEqual(summary["total"], 3)
            self.assertEqual(summary["passed"], 1)
            self.assertEqual(summary["failed"], 2)
            self.assertEqual(summary["python_runtime"], sys.version.split()[0])
            self.assertEqual(len(report.read_text(encoding="utf-8").splitlines()), 2)
            self.assertEqual(
                json.loads(filtered.read_text(encoding="utf-8"))["commit"], "good"
            )
            self.assertEqual(
                yaml.safe_load(summary_file.read_text(encoding="utf-8")),
                summary,
            )
            self.assertEqual(
                main(
                    [
                        "--input-file", str(source),
                        "--report-file", str(root / "issues-2.jsonl"),
                        "--filtered-file", str(root / "filtered-2.jsonl"),
                        "--summary-file", str(root / "summary-2.yaml"),
                        "--fail-on-issues",
                        "--no-progress",
                    ]
                ),
                1,
            )

            self.assertEqual(
                main(["--input-file", str(source), "--no-progress"]),
                0,
            )
            self.assertTrue((root / "input_quality_issues.jsonl").is_file())
            self.assertTrue((root / "input_quality_filtered.jsonl").is_file())
            default_summary = root / "input_quality_summary.yaml"
            self.assertTrue(default_summary.is_file())
            default_summary_data = yaml.safe_load(
                default_summary.read_text(encoding="utf-8")
            )
            self.assertEqual(default_summary_data["total"], 3)
            self.assertEqual(
                default_summary_data["summary_file"], str(default_summary)
            )

    def test_custom_instruction_field_filters_empty_instructions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            report = root / "issues.jsonl"
            filtered = root / "filtered.jsonl"
            summary_file = root / "summary.yaml"
            custom_field = "edit_request"
            valid = {
                PRE: "value = 1\n",
                POST: "value = 2\n",
                custom_field: "Increment the value.",
                "commit": "good",
            }
            invalid = {
                PRE: "value = 1\n",
                POST: "value = 2\n",
                custom_field: " \n",
                "commit": "bad",
            }
            source.write_text(
                json.dumps(valid) + "\n" + json.dumps(invalid) + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                main(
                    [
                        "--input-file", str(source),
                        "--report-file", str(report),
                        "--filtered-file", str(filtered),
                        "--summary-file", str(summary_file),
                        "--instruction-field", custom_field,
                        "--fail-on-issues",
                        "--no-progress",
                    ]
                ),
                1,
            )
            summary = yaml.safe_load(summary_file.read_text(encoding="utf-8"))
            self.assertEqual(summary["passed"], 1)
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(summary["issue_counts"], {"empty_instruction": 1})
            self.assertEqual(
                json.loads(filtered.read_text(encoding="utf-8"))["commit"], "good"
            )
            reported = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(reported["commit"], "bad")
            self.assertEqual(reported["issues"][0]["side"], "instruction")
            self.assertEqual(reported["issues"][0]["field"], custom_field)

    def test_rejects_overlapping_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.jsonl"
            source.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                check_jsonl(source, source, Path(directory) / "filtered.jsonl")
            with self.assertRaises(ValueError):
                check_jsonl(
                    source,
                    Path(directory) / "issues.jsonl",
                    Path(directory) / "filtered.jsonl",
                    summary_file=source,
                )


if __name__ == "__main__":
    unittest.main()
