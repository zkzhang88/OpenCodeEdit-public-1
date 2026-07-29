import json
from openai import OpenAI
import datetime
import argparse
import os
import time
import yaml
from pathlib import Path


MAX_RETRIES = 5  # Maximum number of retries
API_KEY_CONFIG_PATH = str(Path(__file__).resolve().parent / "api_keys.yaml")

def load_api_config(api_config_path: str):
    """Load API keys and base URLs from the YAML config file."""

    with open(api_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    required_fields = (
        "QWEN_API_KEY",
        "QWEN_BASE_URL",
        "QWEN_API_MODEL_NAME",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_API_MODEL_NAME",
    )
    missing_fields = [field for field in required_fields if not config.get(field)]
    if missing_fields:
        raise ValueError(
            f"Missing required fields in YAML config file: {', '.join(missing_fields)}"
        )

    return tuple(config[field] for field in required_fields)


(
    QWEN_API_KEY,
    QWEN_BASE_URL,
    QWEN_API_MODEL_NAME,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_API_MODEL_NAME,
) = load_api_config(API_KEY_CONFIG_PATH)


def load_completed_task_ids(output_path, expected_task_ids, model_name):
    """Read completed task IDs and repair only an incomplete final JSONL line."""

    output_path = Path(output_path)
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()

    completed_task_ids = set()
    with open(output_path, 'r+b') as outfile:
        file_size = os.fstat(outfile.fileno()).st_size
        while True:
            line_start = outfile.tell()
            raw_line = outfile.readline()
            if not raw_line:
                break
            is_last_line = outfile.tell() == file_size

            try:
                output_data = json.loads(raw_line.decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                if is_last_line:
                    outfile.seek(line_start)
                    outfile.truncate()
                    outfile.flush()
                    os.fsync(outfile.fileno())
                    break
                raise ValueError(
                    f"Invalid output JSONL record at byte offset {line_start}: {error}"
                ) from error

            required_fields = ('prompt_id', 'sample_index', 'task_id', 'model_name')
            missing_fields = [
                field for field in required_fields if field not in output_data
            ]
            if missing_fields:
                raise ValueError(
                    f"Output record at byte offset {line_start} is missing fields: "
                    f"{', '.join(missing_fields)}"
                )

            prompt_id = output_data['prompt_id']
            sample_index = output_data['sample_index']
            task_id = output_data['task_id']
            if (
                isinstance(prompt_id, bool)
                or not isinstance(prompt_id, int)
                or prompt_id <= 0
                or isinstance(sample_index, bool)
                or not isinstance(sample_index, int)
                or sample_index <= 0
            ):
                raise ValueError(f"Output task {task_id!r} has invalid identity fields.")
            if task_id != f"{prompt_id}:{sample_index}":
                raise ValueError(
                    f"Output task_id {task_id!r} does not match prompt_id and "
                    "sample_index."
                )
            if output_data['model_name'] != model_name:
                raise ValueError(f"Output task {task_id} uses a different model_name.")
            if task_id not in expected_task_ids:
                raise ValueError(f"Output task {task_id} is not present in the input tasks.")
            if task_id in completed_task_ids:
                raise ValueError(f"Duplicate task_id in output: {task_id}")
            completed_task_ids.add(task_id)

            if is_last_line and not raw_line.endswith(b'\n'):
                outfile.seek(0, os.SEEK_END)
                outfile.write(b'\n')
                outfile.flush()
                os.fsync(outfile.fileno())

    return completed_task_ids


 # Process each record and call the API
def api_infer(input_path, output_path, model_name, num_completion=1, max_samples=None, output_fields=None,
                 continue_from_error=False, temperature=0.8, top_p=0.95, max_tokens=2048, debug=False):
    """
    Read records from the input file, call the API for each record to generate instructive text, and write the results to the output file.
    
    Args:
        input_path (str): Input file path
        output_path (str): Output file path
        model_name (str): Model name. Please check the model list at https://help.aliyun.com/zh/model-studio/getting-started/models;
                          For DeepSeek API, use "deepseek-v3".
        num_completion (int): Number of samples generated for each input, default is 1
        max_samples (int): Number of input samples to select, default is None (no limit)
        output_fields (list): List of output fields, default is None (output all fields)
        continue_from_error (bool): Whether to continue from the last error, default is False
        temperature (float): Temperature parameter, controls diversity of generated text, default is 0.8
        top_p (float): Top-p parameter, controls diversity of generated text, default is 0.95
        max_tokens (int): Maximum length of generated text, default is 2048
        debug (bool): Whether to print debug information, default is False
    Returns:
        None
    """

    if num_completion < 1:
        raise ValueError("num_completion must be at least 1.")
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples cannot be negative.")
    if model_name == "qwen3-32b":
        client = OpenAI(
            api_key=QWEN_API_KEY,
            base_url=QWEN_BASE_URL,
        )
        api_model_name = QWEN_API_MODEL_NAME
        extra_body = {"enable_thinking": False}  # Disable thinking mode
    elif model_name == "deepseek-v3":
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
        api_model_name = DEEPSEEK_API_MODEL_NAME
        extra_body = {}
    else:
        raise ValueError(f"Unsupported model_name: {model_name}. Please use 'qwen3-32b' or 'deepseek-v3'.")

    # Print all the hyperparameters
    print(f"Model: {model_name}")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Continue from error: {continue_from_error}")
    print(f"Number of completions: {num_completion}, Max samples: {max_samples}," 
          f"Temperature: {temperature}, Top-p: {top_p}, Max tokens: {max_tokens}")
    
    if not continue_from_error and Path(output_path).exists() and Path(output_path).stat().st_size > 0:
        raise ValueError(
            f"Output file is not empty: {output_path}. Use a new file or set "
            "--continue_from_error."
        )

    # Read and validate all input records before calling the API.
    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    records = []
    prompt_ids = set()
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in {input_path} at line {line_number}: {error}"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(f"Input line {line_number} must contain a JSON object.")

        prompt_id = record.get('prompt_id')
        if (
            isinstance(prompt_id, bool)
            or not isinstance(prompt_id, int)
            or prompt_id <= 0
        ):
            raise ValueError(
                f"Input line {line_number} has no valid positive integer prompt_id. "
                "Regenerate the prompt file with create_prompt.py."
            )
        if prompt_id in prompt_ids:
            raise ValueError(f"Duplicate prompt_id in input: {prompt_id}")
        prompt_ids.add(prompt_id)

        user_content_list = record.get('user')
        if isinstance(user_content_list, str):
            user_content_list = [user_content_list]
        if (
            not isinstance(user_content_list, list)
            or not user_content_list
            or any(
                not isinstance(content, str) or not content.strip()
                for content in user_content_list
            )
        ):
            raise ValueError(
                f"Input prompt {prompt_id} must contain non-empty user messages."
            )
        records.append(record)

    if max_samples is not None:
        selected_records = records[:max_samples]
    else:
        selected_records = records

    expected_task_ids = {
        f"{record['prompt_id']}:{sample_index}"
        for record in selected_records
        for sample_index in range(1, num_completion + 1)
    }
    completed_task_ids = (
        load_completed_task_ids(output_path, expected_task_ids, model_name)
        if continue_from_error
        else set()
    )
    print(f"There have been {len(completed_task_ids)} records saved to output file before.")

    with open(output_path, 'a', encoding='utf-8') as outfile:
        for i, record in enumerate(selected_records):
            system_content = record.get('system', 'You are a helpful assistant.')
            user_content_list = record.get('user', '')
            if isinstance(user_content_list, str):
                # If user_content_list is a string, convert it to a list
                user_content_list = [user_content_list]

            current_time = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            print("\n")
            print(current_time)
            print(f"Processing input sample {i + 1} of {len(selected_records)}")
            if debug:
                print(f"Input information:\nSystem: {system_content}\nUser: {user_content_list}")

            # Call API
            api_busy = False  # Flag to indicate if API is busy
            for j in range(num_completion):
                sample_index = j + 1
                task_id = f"{record['prompt_id']}:{sample_index}"
                if task_id in completed_task_ids:
                    continue

                input_messages = [{'role': 'system', 'content': system_content}]
                llm_response = []  # Used to store LLM responses for each round
                for round_k, user_input in enumerate(user_content_list):
                    input_messages.append({'role': 'user', 'content': user_input})

                    if debug:
                        print(f"Round {round_k + 1} input messages: {input_messages}")

                    # Call API to generate response
                    completion = client.chat.completions.create(
                        model=api_model_name,
                        messages=input_messages,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        extra_body=extra_body
                    )

                    # Handle API busy situation
                    if completion == "":
                        # Retry 5 times
                        api_busy = True
                        for attempt in range(MAX_RETRIES):
                            time.sleep(10)  # Wait 10 seconds before retrying
                            print(f"API is busy, retrying {attempt + 1}/{MAX_RETRIES}...")
                            completion = client.chat.completions.create(
                                model=api_model_name,
                                messages=input_messages,
                                temperature=temperature,
                                top_p=top_p,
                                max_tokens=max_tokens,
                                extra_body=extra_body
                            )
                            if completion != "":
                                api_busy = False
                                break

                        if api_busy:
                            raise Exception("API is still busy after maximum retries. Please try again later.")
                    
                    llm_response.append(completion.choices[0].message.content)
                    input_messages.append({'role': 'assistant', 'content': completion.choices[0].message.content})  # Add LLM response to input messages

                    if debug:
                        print(f"Round {round_k + 1} response: {llm_response[round_k]}")


                output_data = record.copy()  # Copy the original record
                for k in range(len(llm_response)):
                    # Add each round response to output_data
                    output_data[f'response_{k + 1}'] = llm_response[k]
                output_data['response'] = llm_response  # List of all round responses
                output_data['sample_index'] = sample_index
                output_data['task_id'] = task_id
                output_data['model_name'] = model_name

                # Prepare output result
                if output_fields:
                    # Find fields in output_fields that are missing in output_data
                    missing_fields = [key for key in output_fields if key not in output_data]

                    # If there are missing fields, print warning or raise exception
                    if missing_fields:
                        print(f"\033[91mWarning: The following fields are missing in output_data: {missing_fields}\033[0m")

                    # Retain only the fields specified in output_fields, and re-rank them by the order in output_fields
                    filtered_output_data = {
                        key: output_data[key]
                        for key in output_fields
                        if key in output_data
                    }
                    for key in ('prompt_id', 'sample_index', 'task_id', 'model_name'):
                        filtered_output_data[key] = output_data[key]
                    output_data = filtered_output_data

                if debug:
                    print(f"All fields in output_data: {list(output_data.keys())}")

                if debug:
                    if 'sample_index' in output_data:
                        print(f"sample_index: {output_data['sample_index']}\nOutput: {output_data['response']}")
                    
                # Write result to output file
                outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                outfile.flush()
                os.fsync(outfile.fileno())
                completed_task_ids.add(task_id)

    missing_task_ids = expected_task_ids - completed_task_ids
    if missing_task_ids:
        raise RuntimeError(
            f"Generation finished with {len(missing_task_ids)} missing task IDs."
        )
    print(
        f"Generation complete: total={len(expected_task_ids)}, "
        f"completed={len(completed_task_ids)}, duplicates=0, missing=0."
    )


if __name__ == "__main__":
    # Input and output file paths
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Input file containing structured prompts")
    parser.add_argument("--output_file", type=str, required=True, help="Output file to save generated instructions")
    parser.add_argument("--model_name", type=str, required=True, choices=["qwen3-32b", "deepseek-v3"],
                        help="Model name: 'qwen3-32b' or 'deepseek-v3'")
    parser.add_argument("--continue_from_error", action='store_true', help="Flag to continue from error")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p for sampling")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum number of tokens for generation")
    parser.add_argument("--num_completion", type=int, default=1, help="Number of completions to generate for each prompt")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode for verbose logging")

    args = parser.parse_args()

    model_name = args.model_name

    # Call the function
    api_infer(
        input_path=args.input_file,
        output_path=args.output_file,
        continue_from_error=args.continue_from_error,
        model_name=model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        num_completion=args.num_completion,
        max_samples=args.max_samples,
        debug=args.debug
    )
