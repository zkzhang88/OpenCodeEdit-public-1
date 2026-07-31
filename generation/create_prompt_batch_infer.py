import json
import random
import difflib
from pathlib import Path
import re

from prompts_for_gen import get_prompts


BATCH_MODEL = "deepseek-ai/DeepSeek-V3"
BATCH_URL = "/v1/chat/completions"
BATCH_MAX_TOKENS = 2048
MAX_BATCH_REQUESTS_PER_FILE = 6000


class BatchJsonlWriter:
    """Write batch requests to consecutively numbered JSONL parts."""

    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self.output_file = None
        self.record_count = 0
        self.part_number = 0

    def __enter__(self):
        self._clean_previous_outputs()
        self._open_next_part()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.output_file is not None:
            self.output_file.close()

    def write(self, batch_request):
        if (
            self.record_count > 0
            and self.record_count % MAX_BATCH_REQUESTS_PER_FILE == 0
        ):
            self.output_file.close()
            self._open_next_part()

        self.output_file.write(
            json.dumps(batch_request, ensure_ascii=False) + "\n"
        )
        self.record_count += 1

    def _part_path(self, part_number):
        return self.output_path.with_name(
            f"{self.output_path.stem}_part{part_number:03d}"
            f"{self.output_path.suffix}"
        )

    def _open_next_part(self):
        self.part_number += 1
        self.output_file = open(
            self._part_path(self.part_number),
            "w",
            encoding="utf-8",
        )

    def _clean_previous_outputs(self):
        if self.output_path.is_file():
            self.output_path.unlink()

        part_name_pattern = re.compile(
            rf"^{re.escape(self.output_path.stem)}_part\d+"
            rf"{re.escape(self.output_path.suffix)}$"
        )
        glob_pattern = (
            f"{self.output_path.stem}_part*{self.output_path.suffix}"
        )
        for candidate in self.output_path.parent.glob(glob_pattern):
            if candidate.is_file() and part_name_pattern.fullmatch(candidate.name):
                candidate.unlink()


def build_batch_request(request_number, system_prompt, user_prompt):
    """Build one SiliconFlow Batch Inference request."""
    return {
        "custom_id": f"request-{request_number}",
        "method": "POST",
        "url": BATCH_URL,
        "body": {
            "model": BATCH_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "max_tokens": BATCH_MAX_TOKENS,
        },
    }




def create_prompt(commit_input_path, oneshot_input_path, prompt_version, prompt_output_path,
                  min_snippet_lines=5, max_snippet_lines=15, sample_num=1, random_seed=None):
    """
    Creates a prompt for a given set of input commit lines and a one-shot input file.
    Args:
        commit_input_path (str): Path to the JSONL file, each line representing a commit with keys 'commit', 'old_contents', 'new_contents', and 'message'.
        oneshot_input_path (str): Path to the one-shot input file containing example data.
        prompt_version (str): Version of the prompt template to use.
        prompt_output_path (str): Base path used to name SiliconFlow Batch JSONL parts.
        sample_num (int, optional): Number of samples to generate in total. Defaults to 1.
        random_seed (int, optional): Seed for random number generator to ensure reproducibility. Defaults to None.
    Returns:
        None
    """

    system_prompt, user_prompt_template = get_prompts(prompt_version)

    # Read data from oneshot_input_path
    with open(oneshot_input_path, 'r', encoding='utf-8') as oneshot_file:
        oneshot_lines = oneshot_file.readlines()

    # Random seed
    random.seed(random_seed)

    skipped_records = 0  # Initialize skipped records counter

    def sample_code_snippet(code):
        """
        Randomly sample a code snippet from code, with length between min_snippet_lines and max_snippet_lines.
        """
        lines = code.splitlines()
        total_length = len(lines)
        if total_length < min_snippet_lines:
            raise ValueError(f"Code has fewer lines than the minimum snippet length {min_snippet_lines}.")
        snippet_length = random.randint(min_snippet_lines, min(max_snippet_lines, total_length))
        start_line = random.randint(0, total_length - snippet_length)
        snippet = '\n'.join(lines[start_line:start_line+snippet_length])
        return snippet

    with BatchJsonlWriter(prompt_output_path) as output_writer:
    # Randomly sample code snippets from commit_input_path
        with open(commit_input_path, 'r', encoding='utf-8') as commit_file:
            commit_lines = commit_file.readlines()
            created_prompt_num = 0  # Number of prompts created
            while created_prompt_num < sample_num:
                # Skip flag
                skip_flag = False
                # Randomly select two lines from commit_lines
                selected_commit_lines = random.sample(commit_lines, 2)
                commit_contents = [json.loads(l) for l in selected_commit_lines]

                # Check if content is empty
                for commit_data in commit_contents:
                    # If old_code, new_code, or commit_num is empty, skip this record
                    if not commit_data.get('old_contents', '') or not commit_data.get('new_contents', '') or not commit_data.get('commit', ''):
                        print(f"\033[91mWarning: Missing old/new code or commit number. Skipping this record.\033[0m")
                        skip_flag = True

                if skip_flag:
                    # Skip if any content is empty
                    skipped_records += 1
                    continue

                commit_num = [commit_data['commit'] for commit_data in commit_contents]
                commit_message = [commit_data.get('message', '') for commit_data in commit_contents]

                try:
                    code_snippet = [sample_code_snippet(commit_data['old_contents']) for commit_data in commit_contents]
                except ValueError as e:
                    # If code lines are fewer than min_snippet_lines, skip this record
                    print(f"\033[91mWarning: {e}. Skipping this record.\033[0m")
                    skipped_records += 1
                    continue

                # Randomly select one one-shot data
                oneshot_data = json.loads(random.choice(oneshot_lines))

                # Fill example fields in user_prompt_template
                if prompt_version.startswith('v5'):
                    user_prompt = [user_prompt_template[0].format(
                        code_snippet_1=code_snippet[0],
                        code_snippet_2=code_snippet[1],
                        code_before_shot=oneshot_data['code_before'],
                        desc_instr_shot=oneshot_data['instruct_descriptive'],
                        lazy_instr_shot=oneshot_data['instruct_lazy']
                    ),
                    user_prompt_template[1]]
                else:
                    raise ValueError("Unsupported prompt version")

                batch_request = build_batch_request(
                    created_prompt_num + 1,
                    system_prompt,
                    user_prompt[0],
                )

                output_writer.write(batch_request)

                created_prompt_num += 1

    print(f"Total skipped records: {skipped_records}")  # Print total number of skipped records


def create_prompt_rewrite_commit(commit_input_path, oneshot_input_path, prompt_version, prompt_output_path, shuffle=False, random_seed=None):
    system_prompt, user_prompt_template = get_prompts(prompt_version)

    skipped_records = 0  # Initialize skipped records counter

    # Used to collect all output data
    output_data = []

    # Read data from oneshot_input_path
    with open(oneshot_input_path, 'r', encoding='utf-8') as oneshot_file:
        oneshot_lines = oneshot_file.readlines()

    with open(commit_input_path, 'r', encoding='utf-8') as commit_file:
        for line in commit_file:
            data = json.loads(line)
            commit_num = data.get('commit', '')
            old_code = data.get('old_contents', '')
            new_code = data.get('new_contents', '')
            commit_message = data.get('message', '')

            # If any variable is empty, skip this record
            if not old_code:
                print(f"\033[91mWarning: Old code is empty. Skipping this record.\nCommit number: {commit_num}\033[0m")
                skipped_records += 1
                continue
            if not new_code:
                print(f"\033[91mWarning: New code is empty. Skipping this record.\nCommit number: {commit_num}\033[0m")
                skipped_records += 1
                continue
            if not commit_message:
                print(f"\033[91mWarning: Commit message is empty. Skipping this record.\nCommit number: {commit_num}\033[0m")
                skipped_records += 1
                continue

            # Randomly select one one-shot data
            oneshot_data = json.loads(random.choice(oneshot_lines))

            if prompt_version.startswith('v5.9'):
                # Generate diff
                diff = difflib.unified_diff(old_code.splitlines(), new_code.splitlines(), lineterm='')
                unified_diff = '\n'.join(diff)

                user_prompt = user_prompt_template[0].format(
                    code_before=old_code,
                    code_diff=unified_diff,
                    commit_message=commit_message,
                    desc_instr_shot=oneshot_data['instruct_descriptive'],
                    lazy_instr_shot=oneshot_data['instruct_lazy']
                )
            else:
                raise ValueError("Unsupported prompt version")

            output_data.append((system_prompt, user_prompt))

    # Shuffle feature
    if shuffle:
        random.shuffle(output_data)

    # Write to file
    with BatchJsonlWriter(prompt_output_path) as output_writer:
        for prompt_id, item in enumerate(output_data, start=1):
            batch_request = build_batch_request(prompt_id, item[0], item[1])
            output_writer.write(batch_request)

    print(f"Total skipped records: {skipped_records}")  # Print total number of skipped records


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create prompts for code generation or commit rewriting.")
    parser.add_argument('--prompt_type', type=str, default='code_extend', help="Type of prompt to create: 'code_extend' or 'rewrite_commit'")
    args = parser.parse_args()

    if args.prompt_type == 'code_extend':
        # Create prompts for code extension
        create_prompt(
            commit_input_path='data/commitpackft_python_cleaned.jsonl',
            oneshot_input_path='few-shot/1-shot-prompt_final_chose.jsonl',
            prompt_version='v5.2',
            prompt_output_path='data/prompt_for_syn_batch_infer.jsonl',
            min_snippet_lines=5,
            max_snippet_lines=15,
            sample_num=30000,
            random_seed=42
        )
    elif args.prompt_type == 'rewrite_commit':
        # Create prompts for commit rewriting
        create_prompt_rewrite_commit(
            commit_input_path='data/commitpackft_python_cleaned.jsonl',
            oneshot_input_path='few-shot/1-shot-prompt_final_chose.jsonl',
            prompt_version='v5.9',
            prompt_output_path='data/prompt_rewrite_commit_batch_infer.jsonl',
            random_seed=42
        )
