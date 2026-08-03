import argparse
import json
from pathlib import Path
import random
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


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COMMIT_INPUT_FILE = (
    SCRIPT_DIR / "data" / "commitpackft_python_cleaned.jsonl"
)
DEFAULT_ONESHOT_INPUT_FILE = (
    SCRIPT_DIR / "few-shot" / "1-shot-prompt_final_chose.jsonl"
)
DEFAULT_RANDOM_SEED = 42
DEFAULT_MIN_SNIPPET_LINES = 5
DEFAULT_MAX_SNIPPET_LINES = 15
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


def replay_prompt_metadata(
    max_request_number,
    commit_input_file=DEFAULT_COMMIT_INPUT_FILE,
    oneshot_input_file=DEFAULT_ONESHOT_INPUT_FILE,
    random_seed=DEFAULT_RANDOM_SEED,
    min_snippet_lines=DEFAULT_MIN_SNIPPET_LINES,
    max_snippet_lines=DEFAULT_MAX_SNIPPET_LINES,
):
    """Replay batch prompt sampling and return metadata indexed by request ID."""
    if max_request_number < 1:
        raise BatchResultError("max_request_number must be positive")
    if min_snippet_lines < 1:
        raise BatchResultError("min_snippet_lines must be positive")
    if max_snippet_lines < min_snippet_lines:
        raise BatchResultError(
            "max_snippet_lines must be greater than or equal to "
            "min_snippet_lines"
        )

    commit_input_file = Path(commit_input_file)
    oneshot_input_file = Path(oneshot_input_file)
    try:
        with open(commit_input_file, "r", encoding="utf-8") as input_file:
            commit_lines = input_file.readlines()
    except OSError as error:
        raise BatchResultError(
            f"Cannot open commit input file {commit_input_file}: {error}"
        ) from error
    try:
        with open(oneshot_input_file, "r", encoding="utf-8") as input_file:
            oneshot_lines = input_file.readlines()
    except OSError as error:
        raise BatchResultError(
            f"Cannot open few-shot input file {oneshot_input_file}: {error}"
        ) from error

    if len(commit_lines) < 2:
        raise BatchResultError(
            f"Commit input file must contain at least two records: {commit_input_file}"
        )
    if not oneshot_lines:
        raise BatchResultError(
            f"Few-shot input file is empty: {oneshot_input_file}"
        )

    rng = random.Random(random_seed)
    metadata = {}
    while len(metadata) < max_request_number:
        selected_lines = rng.sample(commit_lines, 2)
        try:
            commit_records = [json.loads(line) for line in selected_lines]
        except (json.JSONDecodeError, TypeError) as error:
            raise BatchResultError(
                f"Cannot replay prompts from {commit_input_file}: {error}"
            ) from error

        if any(
            not record.get("old_contents", "")
            or not record.get("new_contents", "")
            or not record.get("commit", "")
            for record in commit_records
        ):
            continue

        code_snippets = []
        try:
            for record in commit_records:
                code_lines = record["old_contents"].splitlines()
                if len(code_lines) < min_snippet_lines:
                    raise ValueError
                snippet_length = rng.randint(
                    min_snippet_lines,
                    min(max_snippet_lines, len(code_lines)),
                )
                start_line = rng.randint(0, len(code_lines) - snippet_length)
                code_snippets.append(
                    "\n".join(
                        code_lines[start_line : start_line + snippet_length]
                    )
                )
        except ValueError:
            continue

        try:
            json.loads(rng.choice(oneshot_lines))
        except (json.JSONDecodeError, TypeError) as error:
            raise BatchResultError(
                f"Cannot replay prompts from {oneshot_input_file}: {error}"
            ) from error

        commits = [record["commit"] for record in commit_records]
        if not all(isinstance(commit, str) and commit for commit in commits):
            raise BatchResultError(
                "Replayed commit metadata must contain two non-empty strings"
            )
        request_number = len(metadata) + 1
        metadata[request_number] = {
            "commit": commits,
            "code_snippet": code_snippets,
        }

    return metadata


def restore_request_commits(requests, replayed_metadata):
    """Validate replayed snippets and attach commit lists to batch requests."""
    for request_number, request in requests.items():
        metadata = replayed_metadata.get(request_number)
        custom_id = f"request-{request_number}"
        if metadata is None:
            raise BatchResultError(f"No replayed metadata found for {custom_id}")

        actual_snippets = request.get("code_snippet")
        expected_snippets = metadata["code_snippet"]
        if not isinstance(actual_snippets, list) or [
            snippet.strip() for snippet in actual_snippets
        ] != [snippet.strip() for snippet in expected_snippets]:
            raise BatchResultError(
                f"Replayed code snippets do not match the request for {custom_id}; "
                "check the commit input, few-shot input, sampling limits, and seed"
            )

        request["commit"] = list(metadata["commit"])


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


def extract_batch_instruct(
    results_dir,
    output_file,
    requests_dir=None,
    commit_input_file=DEFAULT_COMMIT_INPUT_FILE,
    oneshot_input_file=DEFAULT_ONESHOT_INPUT_FILE,
    random_seed=DEFAULT_RANDOM_SEED,
    min_snippet_lines=DEFAULT_MIN_SNIPPET_LINES,
    max_snippet_lines=DEFAULT_MAX_SNIPPET_LINES,
):
    results_dir = Path(results_dir)
    output_file = Path(output_file)
    if requests_dir is None:
        requests_dir = infer_requests_dir(results_dir)

    requests = load_second_round_requests(requests_dir)
    replayed_metadata = replay_prompt_metadata(
        max(requests),
        commit_input_file=commit_input_file,
        oneshot_input_file=oneshot_input_file,
        random_seed=random_seed,
        min_snippet_lines=min_snippet_lines,
        max_snippet_lines=max_snippet_lines,
    )
    restore_request_commits(requests, replayed_metadata)
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
    parser.add_argument(
        "--commit-input-file",
        type=Path,
        default=DEFAULT_COMMIT_INPUT_FILE,
        help=f"Source commit JSONL file (default: {DEFAULT_COMMIT_INPUT_FILE})",
    )
    parser.add_argument(
        "--oneshot-input-file",
        type=Path,
        default=DEFAULT_ONESHOT_INPUT_FILE,
        help=f"Few-shot JSONL file (default: {DEFAULT_ONESHOT_INPUT_FILE})",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Prompt sampling seed (default: {DEFAULT_RANDOM_SEED})",
    )
    parser.add_argument(
        "--min-snippet-lines",
        type=int,
        default=DEFAULT_MIN_SNIPPET_LINES,
        help=(
            "Minimum sampled code snippet length "
            f"(default: {DEFAULT_MIN_SNIPPET_LINES})"
        ),
    )
    parser.add_argument(
        "--max-snippet-lines",
        type=int,
        default=DEFAULT_MAX_SNIPPET_LINES,
        help=(
            "Maximum sampled code snippet length "
            f"(default: {DEFAULT_MAX_SNIPPET_LINES})"
        ),
    )
    return parser


def main():
    args = build_argument_parser().parse_args()
    result_count = extract_batch_instruct(
        args.results_dir,
        args.output_file,
        requests_dir=args.requests_dir,
        commit_input_file=args.commit_input_file,
        oneshot_input_file=args.oneshot_input_file,
        random_seed=args.random_seed,
        min_snippet_lines=args.min_snippet_lines,
        max_snippet_lines=args.max_snippet_lines,
    )
    print(
        f"Processed {result_count} batch results in custom_id order. "
        f"Output written to {args.output_file}."
    )


if __name__ == "__main__":
    main()
