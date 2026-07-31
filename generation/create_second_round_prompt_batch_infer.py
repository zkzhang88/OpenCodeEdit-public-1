import argparse
from copy import deepcopy
import json
from pathlib import Path
import re

if __package__:
    from .prompts_for_gen import get_prompts
else:
    from prompts_for_gen import get_prompts


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FIRST_ROUND_BASE_FILE = (
    SCRIPT_DIR / "data" / "prompt_for_syn_batch_infer.jsonl"
)
DEFAULT_RESULTS_DIR = (
    SCRIPT_DIR / "data" / "prompt_for_syn_batch_infer_results"
)
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR / "data" / "prompt_for_syn_batch_infer_round2"
)
OUTPUT_STEM = "prompt_for_syn_batch_infer_round2"
MAX_BATCH_REQUESTS_PER_FILE = 6000
CUSTOM_ID_PATTERN = re.compile(r"^request-([1-9]\d*)$")


class BatchConversionError(ValueError):
    """Raised when first-round batch data cannot be converted safely."""


def parse_custom_id(custom_id, context):
    """Return the numeric portion of a canonical ``request-N`` custom ID."""
    if not isinstance(custom_id, str):
        raise BatchConversionError(f"{context}: custom_id must be a string")

    match = CUSTOM_ID_PATTERN.fullmatch(custom_id)
    if match is None:
        raise BatchConversionError(
            f"{context}: invalid custom_id {custom_id!r}; "
            "expected request-<positive integer>"
        )
    return int(match.group(1))


def read_jsonl(path):
    """Yield ``(line_number, record)`` pairs from a strict JSONL file."""
    try:
        input_file = open(path, "r", encoding="utf-8")
    except OSError as error:
        raise BatchConversionError(f"Cannot open {path}: {error}") from error

    with input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                raise BatchConversionError(
                    f"{path}:{line_number}: blank lines are not valid JSONL records"
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise BatchConversionError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise BatchConversionError(
                    f"{path}:{line_number}: JSONL record must be an object"
                )
            yield line_number, record


def discover_first_round_files(first_round_base_file):
    """Find first-round part files, including retry suffixes such as ``_1``."""
    base_path = Path(first_round_base_file)
    part_pattern = re.compile(
        rf"^{re.escape(base_path.stem)}_part\d+(?:_\d+)?"
        rf"{re.escape(base_path.suffix)}$"
    )
    candidates = [
        path
        for path in base_path.parent.glob(
            f"{base_path.stem}_part*{base_path.suffix}"
        )
        if path.is_file() and part_pattern.fullmatch(path.name)
    ]
    if not candidates:
        raise BatchConversionError(
            "No first-round batch files found for "
            f"{base_path.parent / (base_path.stem + '_part*' + base_path.suffix)}"
        )
    return sorted(candidates)


def validate_first_round_request(record, context):
    """Validate the fields needed to reconstruct a complete conversation."""
    custom_id = record.get("custom_id")
    request_number = parse_custom_id(custom_id, context)

    if not isinstance(record.get("method"), str) or not record["method"]:
        raise BatchConversionError(f"{context}: missing or invalid method")
    if not isinstance(record.get("url"), str) or not record["url"]:
        raise BatchConversionError(f"{context}: missing or invalid url")

    body = record.get("body")
    if not isinstance(body, dict):
        raise BatchConversionError(f"{context}: body must be an object")
    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise BatchConversionError(
            f"{context}: first-round messages must contain exactly "
            "one system message and one user message"
        )

    expected_roles = ("system", "user")
    for index, (message, expected_role) in enumerate(
        zip(messages, expected_roles)
    ):
        if not isinstance(message, dict):
            raise BatchConversionError(
                f"{context}: messages[{index}] must be an object"
            )
        if message.get("role") != expected_role:
            raise BatchConversionError(
                f"{context}: messages[{index}].role must be {expected_role!r}"
            )
        if not isinstance(message.get("content"), str) or not message["content"]:
            raise BatchConversionError(
                f"{context}: messages[{index}].content must be a non-empty string"
            )

    return request_number, custom_id


def load_first_round_requests(first_round_base_file):
    """Index first-round requests by numeric custom ID."""
    requests_by_number = {}
    source_by_number = {}

    for path in discover_first_round_files(first_round_base_file):
        for line_number, record in read_jsonl(path):
            context = f"{path}:{line_number}"
            request_number, custom_id = validate_first_round_request(
                record, context
            )
            existing = requests_by_number.get(request_number)
            if existing is not None:
                if existing != record:
                    raise BatchConversionError(
                        f"{context}: conflicting first-round request for "
                        f"{custom_id}; first seen at "
                        f"{source_by_number[request_number]}"
                    )
                continue

            requests_by_number[request_number] = record
            source_by_number[request_number] = context

    return requests_by_number


def discover_result_files(results_dir):
    """Find first-round result JSONL files."""
    results_path = Path(results_dir)
    if not results_path.is_dir():
        raise BatchConversionError(
            f"Results directory does not exist: {results_path}"
        )
    result_files = sorted(
        path for path in results_path.glob("*.jsonl") if path.is_file()
    )
    if not result_files:
        raise BatchConversionError(
            f"No JSONL result files found in {results_path}"
        )
    return result_files


def validate_result_record(record, context):
    """Extract one valid assistant response from a batch result record."""
    custom_id = record.get("custom_id")
    request_number = parse_custom_id(custom_id, context)

    if record.get("error") is not None:
        raise BatchConversionError(
            f"{context}: batch response for {custom_id} contains an error: "
            f"{record['error']!r}"
        )

    response = record.get("response")
    if not isinstance(response, dict):
        raise BatchConversionError(
            f"{context}: response for {custom_id} must be an object"
        )
    body = response.get("body")
    if not isinstance(body, dict):
        raise BatchConversionError(
            f"{context}: response.body for {custom_id} must be an object"
        )
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise BatchConversionError(
            f"{context}: response for {custom_id} must contain exactly one choice"
        )

    choice = choices[0]
    if not isinstance(choice, dict):
        raise BatchConversionError(
            f"{context}: choice for {custom_id} must be an object"
        )
    message = choice.get("message")
    if not isinstance(message, dict):
        raise BatchConversionError(
            f"{context}: assistant message for {custom_id} must be an object"
        )
    if message.get("role") != "assistant":
        raise BatchConversionError(
            f"{context}: response message role for {custom_id} "
            "must be 'assistant'"
        )
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise BatchConversionError(
            f"{context}: assistant content for {custom_id} "
            "must be a non-empty string"
        )

    assistant_message = {"role": "assistant", "content": content}
    return request_number, custom_id, assistant_message


def load_result_batches(results_dir):
    """Load, validate, and numerically order result files and their records."""
    batches = []
    result_source_by_number = {}

    for path in discover_result_files(results_dir):
        records = []
        for line_number, record in read_jsonl(path):
            context = f"{path}:{line_number}"
            request_number, custom_id, assistant_message = (
                validate_result_record(record, context)
            )
            if request_number in result_source_by_number:
                raise BatchConversionError(
                    f"{context}: duplicate result for {custom_id}; first seen at "
                    f"{result_source_by_number[request_number]}"
                )
            result_source_by_number[request_number] = context
            records.append(
                (request_number, custom_id, assistant_message)
            )

        if not records:
            raise BatchConversionError(f"{path}: result file is empty")
        if len(records) > MAX_BATCH_REQUESTS_PER_FILE:
            raise BatchConversionError(
                f"{path}: contains {len(records)} records; maximum batch size is "
                f"{MAX_BATCH_REQUESTS_PER_FILE}"
            )

        records.sort(key=lambda item: item[0])
        batches.append(
            {
                "path": path,
                "first_number": records[0][0],
                "last_number": records[-1][0],
                "records": records,
            }
        )

    batches.sort(key=lambda batch: batch["first_number"])
    for previous, current in zip(batches, batches[1:]):
        if previous["last_number"] >= current["first_number"]:
            raise BatchConversionError(
                "Result file custom_id ranges overlap or interleave: "
                f"{previous['path']} "
                f"(request-{previous['first_number']}.."
                f"request-{previous['last_number']}) and "
                f"{current['path']} "
                f"(request-{current['first_number']}.."
                f"request-{current['last_number']})"
            )

    return batches


def build_second_round_request(
    first_round_request, assistant_message, second_round_prompt
):
    """Append the first response and v5.2 follow-up to the original request."""
    second_round_request = deepcopy(first_round_request)
    original_messages = second_round_request["body"]["messages"]
    second_round_request["body"]["messages"] = [
        *original_messages,
        deepcopy(assistant_message),
        {"role": "user", "content": second_round_prompt},
    ]
    return second_round_request


def output_part_path(output_dir, part_number):
    """Return the required output filename for a numbered result batch."""
    return Path(output_dir) / f"{OUTPUT_STEM}_part{part_number:03d}.jsonl"


def write_output_batches(
    batches, first_round_requests, second_round_prompt, output_dir
):
    """Write one second-round part for each ordered first-round result file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_pattern = re.compile(
        rf"^{re.escape(OUTPUT_STEM)}_part\d+\.jsonl$"
    )
    old_outputs = {
        path
        for path in output_path.glob(f"{OUTPUT_STEM}_part*.jsonl")
        if path.is_file() and output_pattern.fullmatch(path.name)
    }

    summaries = []
    temporary_outputs = []
    try:
        for part_number, batch in enumerate(batches, start=1):
            final_path = output_part_path(output_path, part_number)
            temporary_path = final_path.with_name(f".{final_path.name}.tmp")
            temporary_outputs.append(temporary_path)

            with open(temporary_path, "w", encoding="utf-8") as output_file:
                for request_number, custom_id, assistant_message in batch[
                    "records"
                ]:
                    first_round_request = first_round_requests.get(
                        request_number
                    )
                    if first_round_request is None:
                        raise BatchConversionError(
                            f"No first-round request found for {custom_id}"
                        )
                    output_record = build_second_round_request(
                        first_round_request,
                        assistant_message,
                        second_round_prompt,
                    )
                    output_file.write(
                        json.dumps(output_record, ensure_ascii=False) + "\n"
                    )

            summaries.append(
                {
                    "path": final_path,
                    "count": len(batch["records"]),
                    "first_custom_id": batch["records"][0][1],
                    "last_custom_id": batch["records"][-1][1],
                }
            )

        final_paths = {summary["path"] for summary in summaries}
        for stale_path in old_outputs - final_paths:
            stale_path.unlink()
        for summary, temporary_path in zip(summaries, temporary_outputs):
            temporary_path.replace(summary["path"])
    finally:
        for temporary_path in temporary_outputs:
            if temporary_path.exists():
                temporary_path.unlink()

    return summaries


def create_second_round_batches(
    first_round_base_file=DEFAULT_FIRST_ROUND_BASE_FILE,
    results_dir=DEFAULT_RESULTS_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    """Convert first-round batch responses into v5.2 second-round requests."""
    result_batches = load_result_batches(results_dir)
    first_round_requests = load_first_round_requests(first_round_base_file)

    missing_custom_ids = [
        custom_id
        for batch in result_batches
        for request_number, custom_id, _ in batch["records"]
        if request_number not in first_round_requests
    ]
    if missing_custom_ids:
        preview = ", ".join(missing_custom_ids[:10])
        if len(missing_custom_ids) > 10:
            preview += f", ... ({len(missing_custom_ids)} total)"
        raise BatchConversionError(
            f"Missing first-round requests for: {preview}"
        )

    _, user_prompts = get_prompts("v5.2")
    if len(user_prompts) < 2 or not isinstance(user_prompts[1], str):
        raise BatchConversionError(
            "Prompt version v5.2 does not define a second-round user prompt"
        )

    return write_output_batches(
        result_batches,
        first_round_requests,
        user_prompts[1],
        output_dir,
    )


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Create v5.2 second-round SiliconFlow Batch Inference requests "
            "from first-round result files."
        )
    )
    parser.add_argument(
        "--first-round-base-file",
        type=Path,
        default=DEFAULT_FIRST_ROUND_BASE_FILE,
        help=(
            "Base first-round path used to discover <stem>_part*.jsonl "
            f"(default: {DEFAULT_FIRST_ROUND_BASE_FILE})"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=(
            "Directory containing result JSONL files "
            f"(default: {DEFAULT_RESULTS_DIR})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for second-round part files (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser


def main():
    args = build_argument_parser().parse_args()
    summaries = create_second_round_batches(
        first_round_base_file=args.first_round_base_file,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
    )

    total_count = sum(summary["count"] for summary in summaries)
    print(
        f"Created {len(summaries)} second-round batch files "
        f"with {total_count} requests."
    )
    for summary in summaries:
        print(
            f"{summary['path']}: {summary['count']} requests "
            f"({summary['first_custom_id']}..{summary['last_custom_id']})"
        )


if __name__ == "__main__":
    main()
