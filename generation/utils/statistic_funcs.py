# -*- coding: utf-8 -*-

import os
import json
import matplotlib.pyplot as plt
import logging
import numpy as np
from collections import Counter
import argparse
from tqdm import tqdm
import difflib
import json
import spacy
import pandas as pd
from collections import Counter
import plotly.express as px

from utils.load_instruct_from_file import load_instructions_from_jsonl
from utils.code_splitter import edit_instruction_splitter

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logging.getLogger('gensim').setLevel(logging.WARNING)


CODE_STOP_WORDS = set([
        'def', 'return', 'import', 'from', 'as', 'class', 'if', 'else',
        'elif', 'for', 'while', 'in', 'break', 'continue', 'None', 'True',
        'False', 'try', 'except', 'finally', 'with', 'pass', 'print',
        'self', 'assert', 'yield', 'global', 'lambda', 'nonlocal',
        'and', 'or', 'not', 'is', '=', '==', '!=', '===', '!==', '>', '<', '>=', 
        '<=', '+', '-', '*', '/', '%', '**', '//', '&', '|', '^', '~', '<<', '>>',
        '+=', '-=', '*=', '/=', '%=', '**=', '//=', '&=', '|=', '^=', '~=', '<<=', '>>=',
        '(', ')', '[', ']', '{', '}', ',', '.', ':', ';',
    ])


def filter_by_modify_lines(data_list, max_modify_lines=70, max_hunk_num=7):
    """
    Filters a list of items based on the number of modified lines between 'old_code' and 'new_code'.
    Each item in `data_list` should be a dictionary containing 'old_code' and 'new_code' keys.
    The function uses `diff_analysis` to compute the number of modified, added, and removed lines.
    Only items with a total number of modified lines greater than 0 and less than or equal to `max_modify_lines` are retained.
    Args:
        data_list (list): List of dictionaries, each containing 'old_code' and 'new_code'.
        max_modify_lines (int, optional): Maximum allowed number of modified lines. Defaults to 70.
    Returns:
        list: Filtered list of items meeting the modification criteria.
    """

    filtered = []
    for item in data_list:
        old_code = item.get('code_before_purify', '')
        new_code = item.get('code_after_purify', '')
        diff_stats = diff_analysis(old_code, new_code)
        modify_lines = diff_stats["modified"] + diff_stats["added"] + diff_stats["removed"]
        hunk_num = diff_stats["hunk_num"]
        if 0 < modify_lines <= max_modify_lines and hunk_num <= max_hunk_num:
            filtered.append(item)

    logging.info(f"Number of items after filtering by diff: {len(filtered)}")
    return filtered


def filter_diff_by_percentile(data_list, rm_percentile=0.5, keep_percentile=None):
    """
    Filters out data at the tail end by percentile based on number of modified lines.
    For example, if rm_percentile=0.5, sorts data by modified lines and removes the bottom 0.5% of samples.
    
    Args:
        data_list (list): List of dictionaries, each containing 'code_before_purify' and 'code_after_purify'.
        rm_percentile (float, optional): Percentage of data to filter out from the tail (0.0-100.0). 
                                        Data is sorted by modified lines (ascending), and the bottom percentile% is removed.
                                        Defaults to 0.5 (removes bottom 0.5% of samples).
        keep_percentile (float, optional): Percentage of data to keep from the head (0.0-100.0).
                                        Data is sorted by modified lines (ascending), and only the top keep_percentile% is kept.
                                        If set, rm_percentile is ignored. Defaults to None.
    
    Returns:
        list: Filtered list with tail percentile% of samples removed (based on modified lines ranking).
    """
    
    if not data_list:
        return []
    
    # Calculate modified lines for each item
    modify_lines_list = []
    for item in data_list:
        old_code = item.get('code_before_purify', '')
        new_code = item.get('code_after_purify', '')
        diff_stats = diff_analysis(old_code, new_code)
        modify_lines = diff_stats["modified"] + diff_stats["added"] + diff_stats["removed"]
        modify_lines_list.append(modify_lines)
    
    # Create (item, modify_lines) pairs and sort by modified lines (ascending)
    data_with_lines = list(zip(data_list, modify_lines_list))
    data_with_lines.sort(key=lambda x: x[1])
    
    # Calculate how many samples to keep (remove bottom percentile%)
    total_count = len(data_list)
    if keep_percentile is not None:
        keep_count = int(total_count * (keep_percentile / 100))
    else:
        keep_count = int(total_count * (1 - rm_percentile / 100))
    
    # Keep the first keep_count samples (those with fewer modified lines)
    filtered_data = [item for item, _ in data_with_lines[:keep_count]]
    
    if keep_count > 0:
        threshold = data_with_lines[keep_count - 1][1]
    else:
        threshold = 0
    
    logging.info(f"Filtered by percentile {rm_percentile}%: {len(data_list)} → {len(filtered_data)} items")
    logging.info(f"Modified lines threshold (last kept sample): {threshold}")
    
    return filtered_data


def hdp_topic_analysis(jsonl_path, field_name, data_format, refit=False, debug=False, random_seed=None, **kwargs):
    """
    Performs Hierarchical Dirichlet Process (HDP) topic modeling analysis on a dataset of instructions.
    This function loads instruction data from a JSONL file, preprocesses the text by splitting into code and word tokens,
    removes stop words and non-identifiers, and then applies HDP topic modeling using Gensim to discover latent topics.
    Args:
        jsonl_path (str): Path to the JSONL file containing instruction data.
        field_name (str): The field name in the JSONL file to extract instructions from.
        data_format (str): The format of the data in the JSONL file.
        debug (bool, optional): If True, enables debug logging for token processing. Defaults to False.
        random_seed (int, optional): Random seed for reproducibility of HDP model. Defaults to None.
    Returns:
        HdpModel: Trained Gensim HDP topic model on the processed instruction data.
    Notes:
        - Requires NLTK and Gensim libraries.
        - Downloads NLTK stopwords and punkt tokenizer if not already present.
        - Assumes existence of CODE_STOP_WORDS, load_instructions_from_jsonl, edit_instruction_splitter, and log objects/functions.
        - Logs the number of topics found and the top 20 topics with their representative words.
    """

    import nltk
    from gensim import corpora
    from gensim.models import HdpModel
    from nltk.corpus import stopwords
    import joblib
    import os

    # Download NLTK stopwords
    nltk.download('punkt')
    nltk.download('stopwords')

    code_stop_words = CODE_STOP_WORDS

    # Get model save path
    base_name = os.path.splitext(os.path.basename(jsonl_path))[0]
    fit_dir = os.path.join(os.path.dirname(__file__), "fit_results")
    os.makedirs(fit_dir, exist_ok=True)
    hdp_model_path = os.path.join(fit_dir, f"{base_name}_hdp_model.joblib")
    hdp_dict_path = os.path.join(fit_dir, f"{base_name}_hdp_dictionary.joblib")
    processed_docs_path = os.path.join(fit_dir, f"{base_name}_hdp_processed_docs.joblib")

    # Load data
    instr_list, _ = load_instructions_from_jsonl(jsonl_path, field_name, data_format)

    # Text preprocessing function
    def preprocess(text):
        stop_words = set(stopwords.words('english'))
        code_tokens, word_tokens = edit_instruction_splitter(text)
        word_tokens = [t for t in word_tokens if t.isalpha()]
        word_tokens = [t for t in word_tokens if t not in stop_words]
        code_tokens = [t for t in code_tokens if t.isidentifier()]
        code_tokens = [t for t in code_tokens if t not in code_stop_words]
        # if debug:
        #     log.info(f"Processed code tokens: {code_tokens}")
        #     log.info(f"Processed word tokens: {word_tokens}")
        return code_tokens + word_tokens

    # If model exists and refit is not required, load directly
    if os.path.exists(hdp_model_path) and os.path.exists(hdp_dict_path) \
            and os.path.exists(processed_docs_path) and not refit:
        log.info(f"Loaded existing HDP model: {hdp_model_path}")
        hdp_model = joblib.load(hdp_model_path)
        dictionary = joblib.load(hdp_dict_path)
        processed_docs = joblib.load(processed_docs_path)
        corpus = [dictionary.doc2bow(doc) for doc in tqdm(processed_docs, desc="Building HDP corpus")]
    else:
        processed_docs = [preprocess(doc) for doc in tqdm(instr_list, desc="Preprocessing instructions for HDP")]
        dictionary = corpora.Dictionary(processed_docs)
        corpus = [dictionary.doc2bow(doc) for doc in tqdm(processed_docs, desc="Building HDP corpus")]
        log.info("Performing HDP topic analysis...")
        hdp_model = HdpModel(corpus=corpus, id2word=dictionary, random_state=random_seed)
        joblib.dump(hdp_model, hdp_model_path)
        joblib.dump(dictionary, hdp_dict_path)
        joblib.dump(processed_docs, processed_docs_path)
        log.info(f"HDP model saved to: {hdp_model_path}")
        log.info(f"HDP dictionary saved to: {hdp_dict_path}")
        log.info(f"Processed docs saved to: {processed_docs_path}")

    # Count hard distribution: dominant topic for each document
    dominant_topics = [max(hdp_model[bow], key=lambda x: x[1])[0] 
                       for bow in tqdm(corpus, desc="Assigning dominant topics (HDP)")]
    topic_counts = Counter(dominant_topics)
    top_topics = topic_counts.most_common(20)
    topic_ids = [tid for tid, _ in top_topics]

    num_topics = len(topic_counts)
    # Get descriptions for these topics
    topic_dict = dict(hdp_model.print_topics(num_topics=num_topics, num_words=10))
    # top_20 = {tid: topic_dict[tid] for tid in topic_ids}

    # Plot topic distribution bar chart (up to 20, sorted by count)
    figure_dir = kwargs.get('figure_dir', None)
    if figure_dir:
        counts = [count for _, count in top_topics]

        os.makedirs(figure_dir, exist_ok=True)
        fig_path_hard = os.path.join(figure_dir, f"{base_name}_hdp_topic_distribution_hard_top20.pdf")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        bars = plt.bar(range(len(topic_ids)), counts, color='mediumseagreen')
        plt.xlabel("Topic ID")
        plt.ylabel("Number of Samples")
        # plt.title(f"{base_name} HDP Top 20 Topic Distribution (Hard)")
        plt.xticks(range(len(topic_ids)), topic_ids)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha='center', va='bottom', fontsize=10)
        plt.savefig(fig_path_hard)
        plt.close()
        log.info(f"HDP hard topic distribution figure saved to: {fig_path_hard}")

    return hdp_model


def filter_data_by_hdp_topic_analysis(jsonl_path, field_name, data_format, max_samples_per_topic=None, max_samples_total=None,
                                      refit=False, debug=False, random_seed=None, output_path=None):
    """
    Perform HDP topic analysis on data, then randomly sample topics with more than max_samples_per_topic samples.
    Args:
        jsonl_path (str): Input JSONL file path
        field_name (str): Field name to extract
        data_format (str): Data format type
        max_samples_per_topic (int): Maximum samples to keep per topic
        refit (bool): Whether to retrain model
        debug (bool): Enable debug mode
        random_seed (int): Random seed
        output_path (str): Output file path, auto-generated if None
    Returns:
        str: Output file path
    """
    import random
    import nltk
    from gensim import corpora
    from gensim.models import HdpModel
    from nltk.corpus import stopwords
    import joblib
    
    # Set random seed
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    if max_samples_total is not None:
        log.info(f"Total sample count set: {max_samples_total}")
    elif max_samples_per_topic is not None:
        log.info(f"Max samples per topic set: {max_samples_per_topic}")
    else:
        log.info("No max sample count set")
        raise ValueError("At least one of max_samples_per_topic or max_samples_total must be set")

    # Download NLTK stopwords
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    
    code_stop_words = CODE_STOP_WORDS
    base_name = os.path.splitext(os.path.basename(jsonl_path))[0]
    
    # 获取模型保存路径
    fit_dir = os.path.join(os.path.dirname(__file__), "fit_results")
    os.makedirs(fit_dir, exist_ok=True)
    hdp_model_path = os.path.join(fit_dir, f"{base_name}_hdp_model.joblib")
    hdp_dict_path = os.path.join(fit_dir, f"{base_name}_hdp_dictionary.joblib")
    processed_docs_path = os.path.join(fit_dir, f"{base_name}_hdp_processed_docs.joblib")
    
    # Load original data
    instr_list, _ = load_instructions_from_jsonl(jsonl_path, field_name, data_format)
    
    # Also read full JSONL data for later filtering
    original_data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                original_data.append(json.loads(line))
    
    log.info(f"Total original data count: {len(original_data)}")
    
    # Text preprocessing function
    def preprocess(text):
        stop_words = set(stopwords.words('english'))
        code_tokens, word_tokens = edit_instruction_splitter(text)
        word_tokens = [t for t in word_tokens if t.isalpha()]
        word_tokens = [t for t in word_tokens if t not in stop_words]
        code_tokens = [t for t in code_tokens if t.isidentifier()]
        code_tokens = [t for t in code_tokens if t not in code_stop_words]
        return code_tokens + word_tokens
    
    # If model exists and refit is not required, load directly
    if os.path.exists(hdp_model_path) and os.path.exists(hdp_dict_path) \
            and os.path.exists(processed_docs_path) and not refit:
        log.info(f"Loaded existing HDP model: {hdp_model_path}")
        hdp_model = joblib.load(hdp_model_path)
        dictionary = joblib.load(hdp_dict_path)
        processed_docs = joblib.load(processed_docs_path)
        corpus = [dictionary.doc2bow(doc) for doc in tqdm(processed_docs, desc="Building HDP corpus")]
    else:
        log.info("Start preprocessing documents...")
        processed_docs = [preprocess(doc) for doc in tqdm(instr_list, desc="Preprocessing instructions for HDP")]
        dictionary = corpora.Dictionary(processed_docs)
        corpus = [dictionary.doc2bow(doc) for doc in tqdm(processed_docs, desc="Building HDP corpus")]
        log.info("Performing HDP topic analysis...")
        hdp_model = HdpModel(corpus=corpus, id2word=dictionary, random_state=random_seed)
        joblib.dump(hdp_model, hdp_model_path)
        joblib.dump(dictionary, hdp_dict_path)
        joblib.dump(processed_docs, processed_docs_path)
    log.info(f"HDP model saved to: {hdp_model_path}")
    
    # Assign dominant topic for each document
    log.info("Assigning dominant topic for each document...")
    dominant_topics = []
    for bow in tqdm(corpus, desc="Assigning dominant topics"):
        topic_probs = hdp_model[bow]
        if topic_probs:
            dominant_topic = max(topic_probs, key=lambda x: x[1])[0]
            dominant_topics.append(dominant_topic)
        else:
            dominant_topics.append(-1)  # No topic assigned
    
    # Count document number for each topic
    topic_counts = Counter(dominant_topics)
    log.info(f"Found {len(topic_counts)} topics")
    
    # Create mapping from topic to document indices
    # 这段代码将每条文档的主导主题编号 dominant_topics 映射到其所在索引，生成字典 topic_to_indices，
    # 其中键为 topic_id，值为包含该主题下所有样本索引的列表，用于后续按主题统计或采样。
    topic_to_indices = {}
    for idx, topic_id in enumerate(dominant_topics):
        topic_to_indices.setdefault(topic_id, []).append(idx)
    
    if max_samples_total:
        sorted_topic_counts = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)

    # Calculate target number to keep for each topic according to max_samples_total
    if max_samples_total is not None:
        total = sum(len(v) for v in topic_to_indices.values())
        log.info(f"Original total sample count: {total}, target total: {max_samples_total}")

        locked = {}
        unlocked = dict(topic_to_indices)

        while True:
            log.info(f"Current locked topics: {len(locked)}, unlocked topics: {len(unlocked)}")
            n_unlocked = len(unlocked)
            if n_unlocked == 0:
                break
            target_per_topic = (max_samples_total - sum(len(v) for v in locked.values())) / n_unlocked

            changed = False
            for topic, indices in list(unlocked.items()):
                log.info(f"Topic {topic}: current sample count {len(indices)}, target sample count {target_per_topic}")
                if len(indices) <= target_per_topic:
                    locked[topic] = indices
                    del unlocked[topic]
                    changed = True
            if not changed:
                break

        quota = (max_samples_total - sum(len(v) for v in locked.values())) / len(unlocked)
        topic_target_counts = {}  # 是一个字典，键为 topic_id，值为该主题最终要保留的样本数（整数）
        for topic, indices in locked.items():
            topic_target_counts[topic] = len(indices)
        for topic in unlocked:
            topic_target_counts[topic] = int(quota)

        diff = max_samples_total - sum(topic_target_counts.values())
        if diff != 0:
            keys = list(unlocked.keys())
            for i in range(abs(diff)):
                topic_target_counts[keys[i % len(keys)]] += (1 if diff > 0 else -1)
    else:
        topic_target_counts = {
            topic: min(len(indices), max_samples_per_topic)
            for topic, indices in topic_to_indices.items()
        }
    
    # 采样 
    filtered_indices = []
    for topic_id, indices in topic_to_indices.items():
        target_n = topic_target_counts[topic_id]
        if len(indices) > target_n:
            sampled_indices = random.sample(indices, target_n)
        else:
            sampled_indices = indices
        filtered_indices.extend(sampled_indices)
    log.info(f"Topic {topic_id}: {len(indices)} → keep {target_n}")
    
    filtered_data = [original_data[idx] for idx in sorted(filtered_indices)]
    
    if output_path is None:
        instruct_gen_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(instruct_gen_dir, "filtered")
        os.makedirs(output_dir, exist_ok=True)
        if max_samples_total is not None:
            suffix = f"total_{max_samples_total}"
        else:
            suffix = f"topic_{max_samples_per_topic}"
        output_path = os.path.join(output_dir, f"{base_name}_topic_sampled_{suffix}.jsonl")
    
    # Ensure directory for output_path exists
    out_dir = os.path.dirname(output_path) or '.'
    os.makedirs(out_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in filtered_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    log.info(f"Filtered data saved to: {output_path}")


def diff_analysis(old_code, new_code, context=3):
    old_lines = old_code.splitlines()
    new_lines = new_code.splitlines()

    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    opcodes = matcher.get_opcodes()
    modified = added = removed = 0
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'replace':
            modified += max(i2 - i1, j2 - j1)
        elif tag == 'insert':
            added += (j2 - j1)
        elif tag == 'delete':
            removed += (i2 - i1)

    # Calculate hunk number
    hunks = []
    current_hunk = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            equal_len = i2 - i1
            if equal_len > 2 * context:
                # If length exceeds 2 * context, split hunk
                if current_hunk:
                    hunks.append(current_hunk)
                    current_hunk = []
                continue
        current_hunk.append((tag, i1, i2, j1, j2))

    if current_hunk:
        hunks.append(current_hunk)

    diff_lines = list(difflib.unified_diff(old_lines, new_lines, n=context))
    hunk_count = sum(1 for line in diff_lines if line.startswith('@@'))

    return {
        'added': added,
        'removed': removed,
        'modified': modified,
        'hunk_num': len(hunks),
        'diff_hunk_num': hunk_count,
    }


def compute_diff_statistics(jsonl_path, figure_dir="statistic_figure", **kwargs):
    """
    Computes statistics and visualizations for code diffs from a JSONL file.
    This function reads a JSONL file containing code diff data, analyzes the differences
    between old and new code using `diff_analysis`, and computes statistics such as the
    minimum, maximum, and median number of modified lines and diff hunks. It also generates
    and saves histograms for the distributions of modified lines and hunk numbers.
    Args:
        jsonl_path (str): Path to the input JSONL file containing code diffs.
        figure_dir (str, optional): Directory to save generated figures. Defaults to "statistic_figure".
        **kwargs: Additional keyword arguments:
            - bin_width_modified (int, optional): Bin width for modified lines histogram. Defaults to 5.
            - bin_width_hunk (int, optional): Bin width for hunk number histogram. Defaults to 1.
    Returns:
        dict: A dictionary containing statistics for modified lines and hunk numbers:
            {
                "modified": {
                    "min": int,  # Minimum number of modified lines
                    "max": int,  # Maximum number of modified lines
                    "median": float,  # Median number of modified lines
                    },
                "hunk_num": {
                    "min": int,  # Minimum number of hunks
                    "max": int,  # Maximum number of hunks
                    "median": float,  # Median number of hunks
                    },
            }

    """
    

    # Read jsonl file
    diff_stats_list = []
    modified_list = []
    hunk_num_list = []
    hunk_num_list_1 = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Analyzing code diffs"):
            data = json.loads(line)
            old_code = data.get("old_code", "")
            new_code = data.get("new_code", "")
            diff_stats = diff_analysis(old_code, new_code)
            diff_stats_list.append(diff_stats)
            modified_list.append(diff_stats["modified"] + diff_stats["added"] + diff_stats["removed"])
            hunk_num_list.append(diff_stats["hunk_num"])
            hunk_num_list_1.append(diff_stats["diff_hunk_num"])

    # Calculate statistics
    def get_stats(arr):
        arr_np = np.array(arr)
        return arr_np.min(), arr_np.max(), np.median(arr_np)

    min_mod, max_mod, median_mod = get_stats(modified_list)
    min_hunk, max_hunk, median_hunk = get_stats(hunk_num_list)
    print(f"[Modified Lines] min: {min_mod}, max: {max_mod}, median: {median_mod}")
    print(f"[Hunk Number] min: {min_hunk}, max: {max_hunk}, median: {median_hunk}")

    # Calculate ratio: modified lines > 70 and hunks > 7
    total = len(modified_list)
    if total > 0:
        mod_gt70_count = sum(1 for m in modified_list if m > 70)
        hunk_gt7_count = sum(1 for h in hunk_num_list if h > 7)
        mod_gt70_ratio = mod_gt70_count / total
        hunk_gt7_ratio = hunk_gt7_count / total
    else:
        mod_gt70_count = hunk_gt7_count = 0
        mod_gt70_ratio = hunk_gt7_ratio = 0.0

    print(f"[Modified Lines > 70] ratio: {mod_gt70_ratio:.2%} ({mod_gt70_count}/{total})")
    print(f"[Hunks > 7] ratio: {hunk_gt7_ratio:.2%} ({hunk_gt7_count}/{total})")

    # Plot distribution
    fig_dir = figure_dir
    bin_width_modified = kwargs.get('bin_width_modified', 5)
    bin_width_hunk = kwargs.get('bin_width_hunk', 1)
    os.makedirs(fig_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(jsonl_path))[0]

    # Modified lines distribution
    plt.figure(figsize=(8, 5))
    modified_bins = np.arange(0, max(modified_list) + bin_width_modified, bin_width_modified)
    plt.hist(modified_list, bins=modified_bins, color="skyblue", edgecolor="black", linewidth=0.3)
    plt.xlabel("Modified Lines")
    plt.ylabel("Number of Samples")
    # plt.title(f"{base_name} Modified Lines Distribution")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{base_name}_modified_lines_hist.pdf"))
    plt.close()

    # Hunk number distribution
    plt.figure(figsize=(8, 5))
    hunk_bins = np.arange(0, max(hunk_num_list) + bin_width_hunk, bin_width_hunk)
    plt.hist(hunk_num_list, bins=hunk_bins, color="salmon", edgecolor="black", linewidth=0.3)
    plt.xlabel("Number of Hunks")
    plt.ylabel("Number of Samples")
    # plt.title(f"{base_name} Hunk Number Distribution")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{base_name}_hunk_num_hist.pdf"))
    plt.close()

    return {
        "modified": {
            "min": min_mod,
            "max": max_mod,
            "median": median_mod,
            "gt70_count": mod_gt70_count,
            "gt70_ratio": mod_gt70_ratio,
            },
        "hunk_num": {
            "min": min_hunk,
            "max": max_hunk,
            "median": median_hunk,
            "gt7_count": hunk_gt7_count,
            "gt7_ratio": hunk_gt7_ratio,
            },
    }


def plot_verb_object_sunburst(
    jsonl_path,
    field_name,
    data_format,
    figure_dir,
    lang: str = "en_core_web_sm",
    top_n_verbs: int = 20,
    top_n_objects_per_verb: int = 10,
    drop_other: bool = True,
    remove_border: bool = True,
    border_width: float = 0.5,
    border_color: str = "white",
):
    # 1. Load NLP model
    try:
        nlp = spacy.load(lang)
    except OSError:
        raise ValueError(f"spaCy model '{lang}' not found. Please run: python -m spacy download {lang}")
    
    # 2. Read jsonl data
    instructions, _ = load_instructions_from_jsonl(jsonl_path, field_name, data_format)

    # 3. Extract verb-object pairs
    pairs = []
    for instr in tqdm(instructions, desc="Extracting verb-object pairs"):
        instr_text = edit_instruction_splitter(instr, tokenize=False)[1]  # Only process natural language part
        doc = nlp(instr_text)
        for token in doc:
            if token.pos_ == "VERB":
                for child in token.children:
                    if child.dep_ in ("dobj", "obj"):  # Direct object
                        verb = token.lemma_.lower()
                        obj = child.lemma_.lower()
                        pairs.append((verb, obj))

    # 4. Count frequency
    pair_counts = Counter(pairs)
    data = [{"verb": v, "object": o, "count": c} for (v, o), c in pair_counts.items()]
    if not data:
        raise ValueError("No verb-object pairs extracted, please check data or language model.")

    df_counts = pd.DataFrame(data)

    # 5. Select top_n_verbs verbs
    verb_freq = df_counts.groupby("verb")["count"].sum().sort_values(ascending=False)
    top_verbs = verb_freq.head(top_n_verbs).index.tolist()
    df_top = df_counts[df_counts["verb"].isin(top_verbs)].copy()

    # 6. For each verb, select top_n_objects_per_verb objects
    filtered_rows = []
    for verb in top_verbs:
        sub = df_top[df_top["verb"] == verb].sort_values("count", ascending=False)
        top_objs = sub.head(top_n_objects_per_verb)
        filtered_rows.append(top_objs)
        if not drop_other and len(sub) > top_n_objects_per_verb:
            other_count = sub.iloc[top_n_objects_per_verb:]["count"].sum()
            if other_count > 0:
                filtered_rows.append(pd.DataFrame({
                    "verb": [verb],
                    "object": ["<other>"],
                    "count": [other_count]
                }))
    df_final = pd.concat(filtered_rows, ignore_index=True)

    # 7. Plot sunburst and save image
    fig = px.sunburst(
        df_final,
        path=["verb", "object"],
        values="count",
        title=f"Top {len(top_verbs)} Verbs & Top {top_n_objects_per_verb} Objects per Verb"
    )
    # 去掉或自定义扇形边框
    if remove_border:
        fig.update_traces(marker=dict(line=dict(width=0)))
    else:
        fig.update_traces(marker=dict(line=dict(width=border_width, color=border_color)))
    fig_dir = figure_dir
    os.makedirs(fig_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(jsonl_path))[0]
    fig_path = os.path.join(fig_dir, f"{base_name}_verb_object_sunburst.png")
    fig.write_image(fig_path)
    print(f"Sunburst plot saved to: {fig_path}")


if __name__ == "__main__":
    from utils.plot_and_save import plot_embedding_scatter, plot_embedding_scatter_with_labels

    # 1. 设置参数
    parser = argparse.ArgumentParser(description="Embedding dimensionality reduction and visualization")
    parser.add_argument("jsonl_path", type=str, help="Path to the JSONL file")
    parser.add_argument("--field_name", type=str, nargs='+', required=True, help="Field names to extract")
    parser.add_argument("--data_format", type=str, required=True, choices=["sharegpt", "general"], help="Data format type (sharegpt or general)")
    parser.add_argument("--data_type", type=str, required=True, help="Embedding data type")
    parser.add_argument("--label_file", type=str, help="Path to CSV file with commit and label columns")
    parser.add_argument("--refit", action="store_true", help="Force refit embeddings and models")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    jsonl_path = args.jsonl_path
    field_name = args.field_name
    data_format = args.data_format
    data_type = args.data_type
    refit = args.refit
    label_file = args.label_file

    figure_dir = os.path.join(os.path.dirname(__file__), "statistic_figure")

    plot_verb_object_sunburst(
        jsonl_path,
        field_name=field_name,
        data_format=data_format,
        figure_dir="statistic_figure/Sunburst",
        top_n_verbs=20,
        top_n_objects_per_verb=10,
        drop_other=True,
        remove_border=True,
    )
