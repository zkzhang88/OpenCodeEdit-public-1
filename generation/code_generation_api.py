import json
from openai import OpenAI
import random
import datetime
import argparse
import time


MAX_RETRIES = 5  # Maximum number of retries

QWEN_API_KEY = "sk-xxxx"  # Replace with your API Key from https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.48eb2bdbvjKMhD&tab=model#/api-key
DEEPSEEK_API_KEY = "sk-xxxx"  # Replace with your API Key from https://platform.deepseek.com/

 # Process each record and call the API
def api_infer(input_path, output_path, recovery_file, model_name, num_completion=1, max_samples=None, output_fields=None,
                 continue_from_error=False, temperature=0.8, top_p=0.95, max_tokens=2048, save_every=1000, random_seed=None, debug=False):
    """
    Read records from the input file, call the API for each record to generate instructive text, and write the results to the output file.
    
    Args:
        input_path (str): Input file path
        output_path (str): Output file path
        recovery_file (str): Recovery file path, used to record unprocessed records for error recovery
        model_name (str): Model name. Please check the model list at https://help.aliyun.com/zh/model-studio/getting-started/models;
                          For DeepSeek API, the model name is "deepseek-chat" or "deepseek-reasoner".
        num_completion (int): Number of samples generated for each input, default is 1
        max_samples (int): Number of input samples to select, default is None (no limit)
        output_fields (list): List of output fields, default is None (output all fields)
        continue_from_error (bool): Whether to continue from the last error, default is False
        temperature (float): Temperature parameter, controls diversity of generated text, default is 0.8
        top_p (float): Top-p parameter, controls diversity of generated text, default is 0.95
        max_tokens (int): Maximum length of generated text, default is 2048
        save_every (int): Save output every n samples, default is 1000
        random_seed (int): Random seed, default is None
        debug (bool): Whether to print debug information, default is False
    Returns:
        None
    """

    if model_name == "qwen3-32b":
        client = OpenAI(
            api_key=QWEN_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        extra_body = {"enable_thinking": False}  # Disable thinking mode
    elif model_name == "deepseek-chat":
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
        )
        extra_body = {}
    else:
        raise ValueError(f"Unsupported model_name: {model_name}. Please use 'qwen3-32b' or 'deepseek-chat'.")

    # Print all the hyperparameters
    print(f"Model: {model_name}")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Recovery path: {recovery_file}")
    print(f"Continue from error: {continue_from_error}")
    print(f"Number of completions: {num_completion}, Max samples: {max_samples}," 
          f"Temperature: {temperature}, Top-p: {top_p}, Max tokens: {max_tokens}")
    
    if continue_from_error:
        # Read data from the temporary file
        with open(recovery_file, 'r', encoding='utf-8') as temp_file:
            selected_lines = temp_file.readlines()
        
    # Read the number of generated data lines from output_path
        with open(output_path, 'r', encoding='utf-8') as outfile:
            generated_lines = outfile.readlines()
        
        save_batch_counter = len(generated_lines)  # Start counting from the number of generated data lines
        
    else:
    # Read all records
        with open(input_path, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()

        if max_samples is not None:
            # Randomly select m records
            max_samples = min(len(lines), max_samples)  # Prevent exceeding file line count
            random.seed(random_seed)  # Fix random seed, if None then not fixed
            selected_lines = random.sample(lines, max_samples)
        else:
            selected_lines = lines  # Select all records

    # Write selected_lines to temporary file for error recovery
        with open(recovery_file, 'w', encoding='utf-8') as temp_file:
            temp_file.writelines(selected_lines)

        save_batch_counter = 0  # Counter for recording the number of generated samples

    print(f"There have been {save_batch_counter} records saved to output file before.")

    with open(output_path, 'a', encoding='utf-8') as outfile:
        remaining_lines = selected_lines.copy() # Make a copy to record remaining unprocessed samples for error recovery
        for i, line in enumerate(selected_lines):
            # Read system and user information from JSONL
            record = json.loads(line)
            system_content = record.get('system', 'You are a helpful assistant.')
            user_content_list = record.get('user', '')
            if isinstance(user_content_list, str):
                # If user_content_list is a string, convert it to a list
                user_content_list = [user_content_list]

            current_time = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            print("\n")
            print(current_time)
            print(f"Processing input sample {i + 1} of {len(selected_lines)}")
            if debug:
                print(f"Input information:\nSystem: {system_content}\nUser: {user_content_list}")

            # Check each entry in user_content_list, skip if any is empty
            if any(not str(content).strip() for content in user_content_list):
                print("\033[91mWarning: One or more user content entries are empty. Skipping this record.\033[0m")
                continue

            # Call API
            api_busy = False  # Flag to indicate if API is busy
            for j in range(num_completion):
                input_messages = [{'role': 'system', 'content': system_content}]
                llm_response = []  # Used to store LLM responses for each round
                for round_k, user_input in enumerate(user_content_list):
                    input_messages.append({'role': 'user', 'content': user_input})

                    if debug:
                        print(f"Round {round_k + 1} input messages: {input_messages}")

                    # Call API to generate response
                    completion = client.chat.completions.create(
                        model=model_name,
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
                                model=model_name,
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
                output_data['sample_index'] = j + 1

                # Prepare output result
                if output_fields:
                    # Find fields in output_fields that are missing in output_data
                    missing_fields = [key for key in output_fields if key not in output_data]

                    # If there are missing fields, print warning or raise exception
                    if missing_fields:
                        print(f"\033[91mWarning: The following fields are missing in output_data: {missing_fields}\033[0m")

                    # Retain only the fields specified in output_fields, and re-rank them by the order in output_fields
                    output_data = {key: output_data[key] for key in output_fields if key in output_data}

                if debug:
                    print(f"All fields in output_data: {list(output_data.keys())}")

                if debug:
                    if 'sample_index' in output_data:
                        print(f"sample_index: {output_data['sample_index']}\nOutput: {output_data['response']}")
                    
                # Write result to output file
                outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                save_batch_counter += 1  # Increment counter for each generated record written

                # Save file every save_every samples, and update temporary file
                if save_batch_counter % save_every == 0:
                    outfile.flush()
                    print(f"Have saved {save_batch_counter} records to {output_path}")

                    # Ensure remaining_lines are written to temporary file after saving generated data
                    with open(recovery_file, 'w', encoding='utf-8') as temp_file:
                        temp_file.writelines(remaining_lines)
            
            # Remove processed line from remaining_lines
            remaining_lines.remove(line)


if __name__ == "__main__":
    # Input and output file paths
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Input file containing structured prompts")
    parser.add_argument("--output_file", type=str, required=True, help="Output file to save generated instructions")
    parser.add_argument("--recovery_file", type=str, required=True, help="File for recovering from error")
    parser.add_argument("--model_name", type=str, required=True, choices=["qwen3-32b", "deepseek-chat"],
                        help="Model name: 'qwen3-32b' or 'deepseek-chat'")
    parser.add_argument("--continue_from_error", action='store_true', help="Flag to continue from error")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p for sampling")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum number of tokens for generation")
    parser.add_argument("--num_completion", type=int, default=1, help="Number of completions to generate for each prompt")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--save_every", type=int, default=20, help="Save output every n samples")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode for verbose logging")

    args = parser.parse_args()

    model_name = args.model_name

    # Call the function
    api_infer(
        input_path=args.input_file,
        output_path=args.output_file,
        recovery_file=args.recovery_file,
        continue_from_error=args.continue_from_error,
        model_name=model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        num_completion=args.num_completion,
        max_samples=args.max_samples,
        save_every=args.save_every,
        random_seed=args.random_seed,
        debug=args.debug
    )
