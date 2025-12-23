import os
import json
import logging
from utils.statistic_funcs import filter_by_modify_lines, filter_diff_by_percentile
from utils.statistic_funcs import filter_data_by_hdp_topic_analysis

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logging.getLogger('gensim').setLevel(logging.WARNING)


def dt_filtering(jsonl_path, field_name, data_format, random_seed=None, filter_settings=None):
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
    Returns:
        None: The function writes filtered data to output files in the "filtered" directory.
    """


    base_name = os.path.splitext(os.path.basename(jsonl_path))[0]
    abs_jsonl_path = os.path.abspath(jsonl_path)
    dir_name = os.path.dirname(abs_jsonl_path)

    max_modify_lines = filter_settings.get("max_modify_lines", 70) if filter_settings else 70
    max_hunk_num = filter_settings.get("max_hunk_num", 7) if filter_settings else 7
    max_samples_total = filter_settings.get("max_samples_total", 10000) if filter_settings else 10000
    refit = filter_settings.get("refit", False) if filter_settings else False


    ### Diff Filtering
    data_list = read_jsonl(jsonl_path)
    # filtered_data = filter_by_modify_lines(data_list, max_modify_lines=max_modify_lines, max_hunk_num=max_hunk_num)
    filtered_data = filter_diff_by_percentile(data_list, rm_percentile=1.0)
    output_filename = f"{base_name}_diff_filtered.jsonl"
    instruct_gen_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(instruct_gen_dir, "data", "filtered")
    os.makedirs(output_dir, exist_ok=True)
    diff_output_path = os.path.join(output_dir, output_filename)
    write_jsonl(filtered_data, diff_output_path)

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
        refit=refit
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
        filter_settings=config.get("filter_settings", None)
    )
