import json
import random
import os
from typing import Optional, List, Union

SYSTEM_PROMPT = "You are a code editor. You will be provided the original code snippet and an instruction that specifies " \
"the changes you need to make. You will produce the changed code, based on the original code and the instruction given. " \
"Only produce the code, do not include any additional prose."


def generate_prompt(input_files: Union[str, List[str]], output_file: str, prompt_format: str = 'share_gpt', random_seed: Optional[int] = None):
    """
    Generates prompts for finetuning datasets from one or multiple input files and writes them to an output file in JSONL format.
    Args:
        input_files (str | list[str]): Path(s) to input JSONL file(s) containing data to be processed. Can be a single path or a list of paths.
        output_file (str): Path to the output file where the constructed prompts will be saved.
        prompt_format (str, optional): Format of the prompt to be constructed. Defaults to 'share_gpt'.
        random_seed (int, optional): Random seed for reproducibility. If provided, sets the seed for random operations (e.g., mixing order).

    Returns:
        None

    """
    if random_seed is not None:
        random.seed(random_seed)  # Set the random seed for reproducibility

    output_dir = os.path.dirname(os.path.abspath(output_file))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Normalize to list of files
    if isinstance(input_files, str):
        files = [input_files]
    elif isinstance(input_files, list):
        files = input_files
    else:
        files = list(input_files)

    constructed_data_all = []
    for in_file in files:
        with open(in_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Construct prompts from the lines of this file
            constructed_data = construct_prompt(
                lines,
                prompt_format=prompt_format,
                code_before_field="code_before_purify",
                instruct_field="instruct_purify",
                code_after_field="code_after_purify",
            )
            constructed_data_all.extend(constructed_data)

    # Mix the constructed data from all files together
    random.shuffle(constructed_data_all)

    # Write the constructed data to the output file in JSONL format
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in constructed_data_all:
            f.write(json.dumps(item) + '\n')


def construct_prompt(input_lines, prompt_format='alpaca', **kwargs):
    """
    Generate prompts from input JSON strings and return the constructed data.

    Args:
        input_lines (list of str): List of JSON strings representing the input lines.
        prompt_format (str): Format of the prompt, either 'alpaca' or 'share_gpt'.
        **kwargs: Additional fields mapping, such as 'code_before_field', 'instruct_field', and 'code_after_field'.

    Returns:
        list of dict: A list of dictionaries containing the constructed prompt data.
    """
    constructed_data = []
    for line in input_lines:
        data = json.loads(line)
        commit = data.get('commit', '')
        code_before = data.get(kwargs.get('code_before_field', 'code_before_purify'), '')
        instruct = data.get(kwargs.get('instruct_field', 'instruct_purify'), '')
        code_after = data.get(kwargs.get('code_after_field', 'code_after_purify'), '')
        codeblock_choice = random.choice(['python', None])

        # Check commit, if it is a list, join as a string
        if isinstance(commit, list):
            commit = ",".join(str(item) for item in commit)

        prompt = edit_prompt(
            old=code_before,
            instr=instruct,
            new="",  # The model will generate the new code
            codeblock_before=codeblock_choice
        )

        # Construct prompt data
        ft_data = format_prompt(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            response=build_response(code_after, codeblock_after="python"),
            prompt_format=prompt_format,
            no_thinking_tag=kwargs.get('no_thinking_tag', False)  # Add tag to avoid thinking mode if specified
        )
        ft_data['commit'] = commit
        ft_data['instr_type'] = data.get('instr_type', 'unknown')
        ft_data['old_code'] = code_before
        ft_data['new_code'] = code_after
        constructed_data.append(ft_data)

    return constructed_data


def format_prompt(system_prompt, user_prompt, response, prompt_format='alpaca', no_thinking_tag=False):
    """
    Format the prompt data based on the specified format.

    Args:
        system_prompt (str): The system-level prompt content.
        user_prompt (str): The user-level prompt content.
        response (str): The model's response content.
        prompt_format (str): The format of the prompt, either 'alpaca' or 'share_gpt'.
        no_thinking_tag (bool): If True, adds a flag to the user prompt indicating no thinking is needed.

    Returns:
        dict: A dictionary containing the formatted prompt data.
    """
    
    if no_thinking_tag:
        user_prompt += " /no_think"  # Add a flag to indicate no thinking is needed

    if prompt_format == 'alpaca':
        return {
            "instruction": user_prompt,
            "system": system_prompt,
            "output": response,
        }
    elif prompt_format == 'share_gpt':
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response},
            ]
        }
    else:
        raise ValueError("Invalid prompt_format. Choose either 'alpaca' or 'share_gpt'.")


def edit_prompt(
    old,
    instr,
    new,
    codeblock_before: Optional[str] = None,
    codeblock_after: Optional[str] = None,
):
    """
    The codeblock_before and codeblock_after arguments are used to specify
    if there should be a codeblock surrounding the code before and after
    the instruction. If None, then no codeblock is used. The string is the
    extension of the codeblock, e.g. "py" or "md".

    The format of the prompt is:
    ## Code Before:
    {old}
    ## Instruction:
    {instr}
    ## Code After:
    {new}

    if the parameter new is None, then the model will generate new code after the ## Code After: line.
    """
    if codeblock_before is not None:
        old = f"```{codeblock_before}\n{old}\n```"
    if codeblock_after is not None:
        new = f"```{codeblock_after}\n{new}\n```"
    before = f"""## Code Before:\n{old}\n"""
    instr = f"""## Instruction:\n{instr}\n"""
    after = f"""## Code After:\n{new}"""
    return before + instr + after


def build_response(response, codeblock_after: Optional[str] = None):
    """
    Builds the response string. If codeblock_after is not None, then the response
    will be wrapped in a codeblock with the specified extension.
    """
    if codeblock_after is not None:
        response = f"```{codeblock_after}\n{response}\n```"
    return response


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate prompts from input JSONL file.")
    parser.add_argument("--input_file", nargs='+', help="Path(s) to the input JSONL file(s).")
    parser.add_argument("--output_file", type=str, help="Path to the output JSONL file.")
    parser.add_argument("--prompt_format", type=str, choices=['alpaca', 'share_gpt'], default='share_gpt', help="Format of the prompt.")
    args = parser.parse_args()

    generate_prompt(input_files=args.input_file, output_file=args.output_file, prompt_format=args.prompt_format)