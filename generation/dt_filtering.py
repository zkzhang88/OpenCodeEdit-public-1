import os
import json
import logging
from utils.statistic_funcs import filter_by_modify_lines
from utils.statistic_funcs import filter_data_by_hdp_topic_analysis
from utils.statistic_funcs import compute_diff_statistics

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logging.getLogger('gensim').setLevel(logging.WARNING)


def dt_filtering(
    jsonl_path,
    field_name,
    data_format,
    random_seed=None,
    filter_settings=None,
    statistics_settings=None,
    run_mode="filter",
):
    """
    Filters and processes data from a JSONL file using diff-based and topic-based criteria.
    This function performs two main filtering steps:
    1. Diff Filtering: Filters data samples based on the number of modified lines and hunks.
    2. HDP Topic Filtering: Further filters the diff-filtered data using Hierarchical Dirichlet Process (HDP) topic analysis.
    Args:
        jsonl_path (str): Path to the input JSONL file containing data samples.
        field_name (str or list): Field name(s) to extract instruction content.
            - For 'sharegpt': a single field specifying the conversation list.
            - For general format: one or two field names to concatenate.
        data_format (str): The construction format of the input dataset ("sharegpt" or "general").
        random_seed (int, optional): Seed for random operations to ensure reproducibility. Defaults to None.
        filter_settings (dict, optional): Dictionary of filtering parameters. Supported keys:
            - "max_modify_lines" (int): Maximum number of modified lines allowed per sample (default: 70).
            - "max_hunk_num" (int): Maximum number of hunks allowed per sample (default: 7).
            - "max_samples_total" (int): Maximum total number of samples after filtering (default: 10000).
            - "refit" (bool): Whether to refit the HDP topic model (default: False).
        statistics_settings (dict, optional): Optional distribution plot settings.
            - "output_diff_distribution" (bool): Plot diff distributions before and after diff filtering.
            - "output_topic_distribution" (bool): Plot topic distributions before and after topic sampling.
            - "output_dir" (str): Directory for generated PDF files.
        run_mode (str): Either "filter" for the complete filtering pipeline or
            "analyze_only" to analyze the original input without filtering it.
    Returns:
        None: Filter mode writes filtered data; analyze-only mode writes only
            the enabled statistics plots and HDP cache artifacts.
    """


    base_name = os.path.splitext(os.path.basename(jsonl_path))[0]
    max_modify_lines = filter_settings.get("max_modify_lines", 70) if filter_settings else 70
    max_hunk_num = filter_settings.get("max_hunk_num", 7) if filter_settings else 7
    max_samples_total = filter_settings.get("max_samples_total", 10000) if filter_settings else 10000
    refit = filter_settings.get("refit", False) if filter_settings else False
    statistics_settings = statistics_settings or {}
    output_diff_distribution = statistics_settings.get("output_diff_distribution", False)
    output_topic_distribution = statistics_settings.get("output_topic_distribution", False)
    statistics_output_dir = statistics_settings.get(
        "output_dir", os.path.join("data", "filtered", "statistics")
    )

    if run_mode not in {"filter", "analyze_only"}:
        raise ValueError(
            f"Unsupported run_mode {run_mode!r}; expected 'filter' or 'analyze_only'"
        )
    if run_mode == "analyze_only" and not (
        output_diff_distribution or output_topic_distribution
    ):
        raise ValueError(
            "analyze_only requires output_diff_distribution or "
            "output_topic_distribution to be enabled"
        )

    if output_diff_distribution or output_topic_distribution:
        os.makedirs(statistics_output_dir, exist_ok=True)

    common_diff_options = {
        "figure_dir": statistics_output_dir,
        "old_code_field": "code_before_purify",
        "new_code_field": "code_after_purify",
    }

    if run_mode == "analyze_only":
        if output_diff_distribution:
            compute_diff_statistics(
                jsonl_path,
                filename_prefix=f"{base_name}_diff_before",
                **common_diff_options,
            )
        if output_topic_distribution:
            filter_data_by_hdp_topic_analysis(
                jsonl_path=jsonl_path,
                field_name=field_name,
                data_format=data_format,
                random_seed=random_seed,
                refit=refit,
                figure_dir=statistics_output_dir,
                figure_base_name=base_name,
                analysis_only=True,
            )
        return


    ### Diff Filtering
    data_list = read_jsonl(jsonl_path)
    filtered_data = filter_by_modify_lines(data_list, max_modify_lines=max_modify_lines, max_hunk_num=max_hunk_num)
    output_filename = f"{base_name}_diff_filtered.jsonl"
    instruct_gen_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(instruct_gen_dir, "data", "filtered")
    os.makedirs(output_dir, exist_ok=True)
    diff_output_path = os.path.join(output_dir, output_filename)
    write_jsonl(filtered_data, diff_output_path)

    if output_diff_distribution:
        compute_diff_statistics(
            jsonl_path,
            filename_prefix=f"{base_name}_diff_before",
            **common_diff_options,
        )
        compute_diff_statistics(
            diff_output_path,
            filename_prefix=f"{base_name}_diff_after",
            **common_diff_options,
        )

    ### HDP Topic Filtering
    output_filename = f"{base_name}_dt_filtered.jsonl"
    output_path = os.path.join(output_dir, output_filename)

    filter_data_by_hdp_topic_analysis(
        jsonl_path=diff_output_path,
        field_name=field_name,
        data_format=data_format,
        output_path=output_path,
        max_samples_total=max_samples_total,
        random_seed=random_seed,
        refit=refit,
        figure_dir=statistics_output_dir if output_topic_distribution else None,
        figure_base_name=base_name,
        analysis_only=False,
    )


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def write_jsonl(data_list, file_path):
    dir_name = os.path.dirname(file_path)
    os.makedirs(dir_name, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Filter dataset using diff and topic modeling.")
    parser.add_argument("--config", type=str, required=True, help="Path to the filter_config.yaml file.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dt_filtering(
        jsonl_path=config["jsonl_path"],
        field_name=config["field_name"],
        data_format=config["data_format"],
        random_seed=config.get("random_seed", None),
        filter_settings=config.get("filter_settings", None),
        statistics_settings=config.get("statistics", None),
        run_mode=config.get("run_mode", "filter"),
    )
