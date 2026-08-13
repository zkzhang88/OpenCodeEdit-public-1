from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


GENERATION_DIR = Path(__file__).resolve().parents[1]
if str(GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATION_DIR))

import dt_filtering as dt_filtering_module
from utils import statistic_funcs


class DtFilteringPlotConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_path = self.root / "sample.jsonl"
        self.statistics_dir = self.root / "statistics"
        self.records = [
            {
                "commit": "one",
                "code_before_purify": "value = 1\n",
                "code_after_purify": "value = 2\n",
                "instruct_purify": "Change the value.",
            }
        ]

    def tearDown(self):
        self.temporary_directory.cleanup()

    def run_filter(self, *, output_diff=False, output_topic=False):
        settings = {
            "output_diff_distribution": output_diff,
            "output_topic_distribution": output_topic,
            "output_dir": str(self.statistics_dir),
        }
        with (
            mock.patch.object(
                dt_filtering_module, "read_jsonl", return_value=self.records
            ),
            mock.patch.object(
                dt_filtering_module,
                "filter_by_modify_lines",
                return_value=self.records,
            ),
            mock.patch.object(dt_filtering_module, "write_jsonl"),
            mock.patch.object(
                dt_filtering_module, "compute_diff_statistics"
            ) as compute_diff,
            mock.patch.object(
                dt_filtering_module, "filter_data_by_hdp_topic_analysis"
            ) as filter_topics,
        ):
            dt_filtering_module.dt_filtering(
                str(self.input_path),
                ["code_before_purify", "instruct_purify"],
                "general",
                random_seed=42,
                statistics_settings=settings,
            )
        return compute_diff, filter_topics

    def test_plot_switches_are_independent(self):
        for output_diff, output_topic in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            with self.subTest(diff=output_diff, topic=output_topic):
                compute_diff, filter_topics = self.run_filter(
                    output_diff=output_diff, output_topic=output_topic
                )
                self.assertEqual(compute_diff.call_count, 2 if output_diff else 0)
                topic_kwargs = filter_topics.call_args.kwargs
                self.assertEqual(
                    topic_kwargs["figure_dir"],
                    str(self.statistics_dir) if output_topic else None,
                )
                self.assertEqual(topic_kwargs["figure_base_name"], "sample")

    def test_diff_calls_use_purified_code_fields_and_expected_prefixes(self):
        compute_diff, _ = self.run_filter(output_diff=True)

        self.assertEqual(
            [call.kwargs["filename_prefix"] for call in compute_diff.call_args_list],
            ["sample_diff_before", "sample_diff_after"],
        )
        for call in compute_diff.call_args_list:
            self.assertEqual(call.kwargs["figure_dir"], str(self.statistics_dir))
            self.assertEqual(call.kwargs["old_code_field"], "code_before_purify")
            self.assertEqual(call.kwargs["new_code_field"], "code_after_purify")

    def test_missing_statistics_configuration_disables_plots(self):
        with (
            mock.patch.object(
                dt_filtering_module, "read_jsonl", return_value=self.records
            ),
            mock.patch.object(
                dt_filtering_module,
                "filter_by_modify_lines",
                return_value=self.records,
            ),
            mock.patch.object(dt_filtering_module, "write_jsonl"),
            mock.patch.object(
                dt_filtering_module, "compute_diff_statistics"
            ) as compute_diff,
            mock.patch.object(
                dt_filtering_module, "filter_data_by_hdp_topic_analysis"
            ) as filter_topics,
        ):
            dt_filtering_module.dt_filtering(
                str(self.input_path),
                ["code_before_purify", "instruct_purify"],
                "general",
            )

        compute_diff.assert_not_called()
        self.assertIsNone(filter_topics.call_args.kwargs["figure_dir"])

    def test_analyze_only_diff_uses_original_input_without_filtering_or_writes(self):
        settings = {
            "output_diff_distribution": True,
            "output_topic_distribution": False,
            "output_dir": str(self.statistics_dir),
        }
        with (
            mock.patch.object(dt_filtering_module, "read_jsonl") as read_jsonl,
            mock.patch.object(
                dt_filtering_module, "filter_by_modify_lines"
            ) as filter_by_diff,
            mock.patch.object(dt_filtering_module, "write_jsonl") as write_jsonl,
            mock.patch.object(
                dt_filtering_module, "compute_diff_statistics"
            ) as compute_diff,
            mock.patch.object(
                dt_filtering_module, "filter_data_by_hdp_topic_analysis"
            ) as filter_topics,
        ):
            dt_filtering_module.dt_filtering(
                str(self.input_path),
                ["code_before_purify", "instruct_purify"],
                "general",
                statistics_settings=settings,
                run_mode="analyze_only",
            )

        read_jsonl.assert_not_called()
        filter_by_diff.assert_not_called()
        write_jsonl.assert_not_called()
        filter_topics.assert_not_called()
        compute_diff.assert_called_once()
        self.assertEqual(compute_diff.call_args.args[0], str(self.input_path))
        self.assertEqual(
            compute_diff.call_args.kwargs["filename_prefix"], "sample_diff_before"
        )

    def test_analyze_only_topic_uses_original_input_without_sampling_or_writes(self):
        settings = {
            "output_diff_distribution": False,
            "output_topic_distribution": True,
            "output_dir": str(self.statistics_dir),
        }
        with (
            mock.patch.object(dt_filtering_module, "read_jsonl") as read_jsonl,
            mock.patch.object(
                dt_filtering_module, "filter_by_modify_lines"
            ) as filter_by_diff,
            mock.patch.object(dt_filtering_module, "write_jsonl") as write_jsonl,
            mock.patch.object(
                dt_filtering_module, "compute_diff_statistics"
            ) as compute_diff,
            mock.patch.object(
                dt_filtering_module, "filter_data_by_hdp_topic_analysis"
            ) as analyze_topics,
        ):
            dt_filtering_module.dt_filtering(
                str(self.input_path),
                ["code_before_purify", "instruct_purify"],
                "general",
                random_seed=42,
                filter_settings={"refit": True},
                statistics_settings=settings,
                run_mode="analyze_only",
            )

        read_jsonl.assert_not_called()
        filter_by_diff.assert_not_called()
        write_jsonl.assert_not_called()
        compute_diff.assert_not_called()
        topic_kwargs = analyze_topics.call_args.kwargs
        self.assertEqual(topic_kwargs["jsonl_path"], str(self.input_path))
        self.assertTrue(topic_kwargs["analysis_only"])
        self.assertTrue(topic_kwargs["refit"])
        self.assertEqual(topic_kwargs["random_seed"], 42)
        self.assertNotIn("output_path", topic_kwargs)
        self.assertNotIn("max_samples_total", topic_kwargs)

    def test_analyze_only_can_enable_both_analyses(self):
        settings = {
            "output_diff_distribution": True,
            "output_topic_distribution": True,
            "output_dir": str(self.statistics_dir),
        }
        with (
            mock.patch.object(dt_filtering_module, "read_jsonl") as read_jsonl,
            mock.patch.object(
                dt_filtering_module, "filter_by_modify_lines"
            ) as filter_by_diff,
            mock.patch.object(dt_filtering_module, "write_jsonl") as write_jsonl,
            mock.patch.object(
                dt_filtering_module, "compute_diff_statistics"
            ) as compute_diff,
            mock.patch.object(
                dt_filtering_module, "filter_data_by_hdp_topic_analysis"
            ) as analyze_topics,
        ):
            dt_filtering_module.dt_filtering(
                str(self.input_path),
                ["code_before_purify", "instruct_purify"],
                "general",
                statistics_settings=settings,
                run_mode="analyze_only",
            )

        read_jsonl.assert_not_called()
        filter_by_diff.assert_not_called()
        write_jsonl.assert_not_called()
        compute_diff.assert_called_once()
        analyze_topics.assert_called_once()

    def test_analyze_only_requires_at_least_one_output(self):
        with self.assertRaisesRegex(ValueError, "analyze_only requires"):
            dt_filtering_module.dt_filtering(
                str(self.input_path),
                ["code_before_purify", "instruct_purify"],
                "general",
                statistics_settings={},
                run_mode="analyze_only",
            )

    def test_invalid_run_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported run_mode"):
            dt_filtering_module.dt_filtering(
                str(self.input_path),
                ["code_before_purify", "instruct_purify"],
                "general",
                run_mode="unknown",
            )


class DistributionPlotTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_diff_statistics_use_configured_fields_and_write_two_pdfs(self):
        input_path = self.root / "records.jsonl"
        input_path.write_text(
            json.dumps(
                {
                    "old_code": "unchanged",
                    "new_code": "unchanged",
                    "code_before_purify": "value = 1",
                    "code_after_purify": "value = 2",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        result = statistic_funcs.compute_diff_statistics(
            str(input_path),
            figure_dir=str(self.root),
            old_code_field="code_before_purify",
            new_code_field="code_after_purify",
            filename_prefix="records_diff_before",
        )

        self.assertEqual(result["modified"]["min"], 1)
        self.assertTrue(
            (self.root / "records_diff_before_modified_lines_hist.pdf").is_file()
        )
        self.assertTrue(
            (self.root / "records_diff_before_hunk_num_hist.pdf").is_file()
        )

    def test_topic_plots_share_assignments_and_use_sampled_indices(self):
        dominant_topics = [4, 4, 7, 9]
        selected_indices = [2, 0]

        with mock.patch.object(
            statistic_funcs, "_plot_topic_distribution"
        ) as plot_distribution:
            statistic_funcs._plot_topic_distributions(
                dominant_topics,
                selected_indices,
                str(self.root),
                "records",
            )

        before_call, after_call = plot_distribution.call_args_list
        self.assertEqual(before_call.args[0], {4: 2, 7: 1, 9: 1})
        self.assertEqual(
            before_call.args[2], "records_topic_before_distribution_top20.pdf"
        )
        self.assertEqual(after_call.args[0], {4: 1, 7: 1})
        self.assertEqual(
            after_call.args[2], "records_topic_after_distribution_top20.pdf"
        )

    def test_topic_plot_helper_writes_two_pdfs(self):
        statistic_funcs._plot_topic_distributions(
            [1, 1, 2], [0, 2], str(self.root), "records"
        )

        self.assertTrue(
            (self.root / "records_topic_before_distribution_top20.pdf").is_file()
        )
        self.assertTrue(
            (self.root / "records_topic_after_distribution_top20.pdf").is_file()
        )

    def test_selected_records_preserve_input_order(self):
        records = [{"id": 0}, {"id": 1}, {"id": 2}]
        selected = statistic_funcs._select_records_in_input_order(records, [2, 0])
        self.assertEqual(selected, [{"id": 0}, {"id": 2}])


if __name__ == "__main__":
    unittest.main()
