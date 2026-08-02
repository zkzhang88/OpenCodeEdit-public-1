import argparse
import json
from pathlib import Path
import re
import tempfile

if __package__:
    from .get_instruct_from_response import (
        filter_singleline_data,
        purify_code_from_jsonl,
        purify_instructions,
        separate_instruct,
    )
else:
    from get_instruct_from_response import (
        filter_singleline_data,
        purify_code_from_jsonl,
        purify_instructions,
        separate_instruct,
    )


CUSTOM_ID_PATTERN = re.compile(r"^request-([1-9]\d*)$")
CODE_SNIPPETS_PATTERN = re.compile(
    r"^## Code Snippet 1:\s*\n"
    r"(?P<first>.*?)"
    r"^## Code Snippet 2:\s*\n"
    r"(?P<second>.*?)"
    r"^## Guidelines for each section:",
    re.MULTILINE | re.DOTALL,
)


class BatchResultError(ValueError):
    """Raised when batch requests and results cannot be matched safely."""


def parse_custom_id(custom_id, context):
    if not isinstance(custom_id, str):
        raise BatchResultError(f"{context}: custom_id must be a string")

    match = CUSTOM_ID_PATTERN.fullmatch(custom_id)
    if match is None:
        raise BatchResultError(
            f"{context}: invalid custom_id {custom_id!r}; "
            "expected request-<positive integer>"
        )
    return int(match.group(1))


def read_jsonl(path):
    try:
        input_file = open(path, "r", encoding="utf-8")
    except OSError as error:
        raise BatchResultError(f"Cannot open {path}: {error}") from error

    with input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                raise BatchResultError(
                    f"{path}:{line_number}: blank lines are not valid JSONL records"
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise BatchResultError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise BatchResultError(
                    f"{path}:{line_number}: JSONL record must be an object"
                )
            yield line_number, record


def discover_jsonl_files(directory, description):
    directory = Path(directory)
    if not directory.is_dir():
        raise BatchResultError(f"{description} directory does not exist: {directory}")

    paths = sorted(path for path in directory.glob("*.jsonl") if path.is_file())
    if not paths:
        raise BatchResultError(f"No JSONL files found in {description} directory: {directory}")
    return paths


def extract_code_snippets(user_prompt, context):
    match = CODE_SNIPPETS_PATTERN.search(user_prompt)
    if match is None:
        raise BatchResultError(
            f"{context}: cannot extract Code Snippet 1 and Code Snippet 2 "
            "from the first-round user prompt"
        )
    return [match.group("first").strip(), match.group("second").strip()]


def load_second_round_requests(requests_dir):
    requests = {}
    sources = {}

    for path in discover_jsonl_files(requests_dir, "second-round request"):
        for line_number, record in read_jsonl(path):
            context = f"{path}:{line_number}"
            request_number = parse_custom_id(record.get("custom_id"), context)
            if request_number in requests:
                raise BatchResultError(
                    f"{context}: duplicate request for {record['custom_id']}; "
                    f"first seen at {sources[request_number]}"
                )

            body = record.get("body")
            messages = body.get("messages") if isinstance(body, dict) else None
            if not isinstance(messages, list) or len(messages) != 4:
                raise BatchResultError(
                    f"{context}: second-round request must contain exactly "
                    "system, user, assistant, and user messages"
                )
            expected_roles = ("system", "user", "assistant", "user")
            if tuple(message.get("role") for message in messages if isinstance(message, dict)) != expected_roles:
                raise BatchResultError(
                    f"{context}: second-round message roles must be "
                    "system, user, assistant, user"
                )

            first_user_prompt = messages[1].get("content")
            response_1 = messages[2].get("content")
            if not isinstance(first_user_prompt, str) or not first_user_prompt:
                raise BatchResultError(
                    f"{context}: first-round user prompt must be a non-empty string"
                )
            if not isinstance(response_1, str) or not response_1.strip():
                raise BatchResultError(
                    f"{context}: first-round assistant response must be non-empty"
                )

            code_snippet = record.get("code_snippet")
            if not (
                isinstance(code_snippet, list)
                and len(code_snippet) == 2
                and all(isinstance(item, str) for item in code_snippet)
            ):
                code_snippet = extract_code_snippets(first_user_prompt, context)

            requests[request_number] = {
                "commit": record.get("commit", ""),
                "code_snippet": code_snippet,
                "user": first_user_prompt,
                "response_1": response_1,
            }
            sources[request_number] = context

    return requests


def extract_response_2(record, context):
    custom_id = record.get("custom_id")
    request_number = parse_custom_id(custom_id, context)

    if record.get("error") is not None:
        raise BatchResultError(
            f"{context}: batch response for {custom_id} contains an error: "
            f"{record['error']!r}"
        )
    response = record.get("response")
    body = response.get("body") if isinstance(response, dict) else None
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or len(choices) != 1:
        raise BatchResultError(
            f"{context}: response for {custom_id} must contain exactly one choice"
        )
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise BatchResultError(
            f"{context}: response message for {custom_id} must have role 'assistant'"
        )
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise BatchResultError(
            f"{context}: assistant response for {custom_id} must be non-empty"
        )
    return request_number, content


def load_batch_results(results_dir):
    results = {}
    sources = {}

    for path in discover_jsonl_files(results_dir, "batch result"):
        for line_number, record in read_jsonl(path):
            context = f"{path}:{line_number}"
            request_number, response_2 = extract_response_2(record, context)
            if request_number in results:
                raise BatchResultError(
                    f"{context}: duplicate result for {record['custom_id']}; "
                    f"first seen at {sources[request_number]}"
                )
            results[request_number] = response_2
            sources[request_number] = context

    return results


def infer_requests_dir(results_dir):
    results_dir = Path(results_dir)
    suffix = "_results"
    if not results_dir.name.endswith(suffix):
        raise BatchResultError(
            "Cannot infer the second-round request directory because the result "
            f"directory name does not end with {suffix!r}; pass --requests-dir"
        )
    return results_dir.with_name(results_dir.name[: -len(suffix)])


def write_ordered_responses(requests, results, output_file):
    request_ids = set(requests)
    result_ids = set(results)
    if request_ids != result_ids:
        missing_results = sorted(request_ids - result_ids)
        missing_requests = sorted(result_ids - request_ids)
        details = []
        if missing_results:
            details.append(
                "requests without results: "
                + ", ".join(f"request-{item}" for item in missing_results[:10])
            )
        if missing_requests:
            details.append(
                "results without requests: "
                + ", ".join(f"request-{item}" for item in missing_requests[:10])
            )
        raise BatchResultError("; ".join(details))

    with open(output_file, "w", encoding="utf-8") as merged_file:
        for request_number in sorted(result_ids):
            record = dict(requests[request_number])
            record["response_2"] = results[request_number]
            merged_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_batch_instruct(results_dir, output_file, requests_dir=None):
    results_dir = Path(results_dir)
    output_file = Path(output_file)
    if requests_dir is None:
        requests_dir = infer_requests_dir(results_dir)

    requests = load_second_round_requests(requests_dir)
    results = load_batch_results(results_dir)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="batch_instruct_") as temp_dir:
        ordered_responses = Path(temp_dir) / "ordered_responses.jsonl"
        separated_responses = Path(temp_dir) / "separated_responses.jsonl"
        filtered_responses = Path(temp_dir) / "filtered_responses.jsonl"
        purified_code = Path(temp_dir) / "purified_code.jsonl"
        write_ordered_responses(requests, results, ordered_responses)
        separate_instruct(
            input_file=str(ordered_responses),
            output_file=str(separated_responses),
        )
        filter_singleline_data(
            input_file=str(separated_responses),
            output_file=str(filtered_responses),
            field_names=["code_before", "code_after"],
        )
        purify_code_from_jsonl(
            input_file=str(filtered_responses),
            output_file=str(purified_code),
            purify_fields=["code_before", "code_after"],
            keep_language_mark=False,
        )
        purify_instructions(
            input_file=str(purified_code),
            output_file=str(output_file),
            purify_fields=["instruct_descriptive", "instruct_lazy"],
        )

    return len(results)


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Extract edit triplets from a directory of second-round batch "
            "inference results, ordered numerically by custom_id."
        )
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing second-round batch result JSONL files",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Path for the extracted triplet JSONL file",
    )
    parser.add_argument(
        "--requests-dir",
        type=Path,
        help=(
            "Directory containing matching second-round request JSONL files "
            "(default: results directory name with trailing '_results' removed)"
        ),
    )
    return parser


def main():
    args = build_argument_parser().parse_args()
    result_count = extract_batch_instruct(
        args.results_dir,
        args.output_file,
        requests_dir=args.requests_dir,
    )
    print(
        f"Processed {result_count} batch results in custom_id order. "
        f"Output written to {args.output_file}."
    )


if __name__ == "__main__":
    main()
