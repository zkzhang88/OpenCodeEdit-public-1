import argparse
from pathlib import Path

API_CONFIG_PATH = Path(__file__).resolve().parent / "api_config.yaml"
DEFAULT_BATCH_ENDPOINT = "/v1/chat/completions"
DEFAULT_COMPLETION_WINDOW = "24h"
DEFAULT_BATCH_MODEL = "deepseek-ai/DeepSeek-V3"


def load_siliconflow_config(api_config_path=API_CONFIG_PATH):
    """Load the SiliconFlow API key and base URL from the local YAML config."""
    import yaml

    try:
        with open(api_config_path, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
    except FileNotFoundError as error:
        raise FileNotFoundError(f"API config file not found: {api_config_path}") from error

    required_fields = ("SILICONFLOW_API_KEY", "SILICONFLOW_BASE_URL")
    missing_fields = [field for field in required_fields if not config.get(field)]
    if missing_fields:
        raise ValueError(
            "Missing required fields in API config: " + ", ".join(missing_fields)
        )

    return config["SILICONFLOW_API_KEY"], config["SILICONFLOW_BASE_URL"]


def upload_batch_file(file_path):
    """Upload a JSONL file for SiliconFlow Batch Inference."""
    from openai import OpenAI

    api_key, base_url = load_siliconflow_config()
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    with open(file_path, "rb") as input_file:
        batch_input_file = client.files.create(
            file=input_file,
            purpose="batch",
        )

    print(batch_input_file)
    file_id = batch_input_file.data["id"]
    print(file_id)
    job_config_path = write_batch_job_config(file_path, file_id)
    print(f"Batch job config written to: {job_config_path}")
    return file_id


def write_batch_job_config(file_path, file_id):
    """Write a ready-to-use batch job YAML next to the uploaded input file."""
    import yaml

    file_path = Path(file_path)
    job_config_path = get_batch_job_config_path(file_path)
    job_config = {
        "file_name": file_path.name,
        "input_file_id": file_id,
        "endpoint": DEFAULT_BATCH_ENDPOINT,
        "completion_window": DEFAULT_COMPLETION_WINDOW,
        "metadata": {
            "description": f"Batch job for {file_path.name}",
        },
        "extra_body": {
            "replace": {
                "model": DEFAULT_BATCH_MODEL,
            }
        },
    }

    with open(job_config_path, "w", encoding="utf-8") as config_file:
        yaml.safe_dump(
            job_config,
            config_file,
            allow_unicode=True,
            sort_keys=False,
        )

    return job_config_path


def get_batch_job_config_path(file_path):
    """Derive the generated batch job config path from an uploaded file path."""
    file_path = Path(file_path)
    return file_path.with_name(f"{file_path.stem}_batch_job.yaml")


def load_batch_job_config(job_config_path):
    """Load and validate arguments for ``client.batches.create``."""
    import yaml

    try:
        with open(job_config_path, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Batch job config file not found: {job_config_path}"
        ) from error

    if not isinstance(config, dict):
        raise ValueError("Batch job config must contain a YAML mapping.")

    required_fields = ("input_file_id", "endpoint", "completion_window")
    missing_fields = [field for field in required_fields if not config.get(field)]
    if missing_fields:
        raise ValueError(
            "Missing required fields in batch job config: "
            + ", ".join(missing_fields)
        )

    api_fields = {
        "input_file_id",
        "endpoint",
        "completion_window",
        "metadata",
        "extra_body",
    }
    supported_fields = api_fields | {"file_name"}
    unsupported_fields = sorted(set(config) - supported_fields)
    if unsupported_fields:
        raise ValueError(
            "Unsupported fields in batch job config: "
            + ", ".join(unsupported_fields)
        )

    return {field: config[field] for field in api_fields if field in config}


def create_batch_job(file_path):
    """Create a batch job using config generated when ``file_path`` was uploaded."""
    from openai import OpenAI

    api_key, base_url = load_siliconflow_config()
    job_config_path = get_batch_job_config_path(file_path)
    job_config = load_batch_job_config(job_config_path)
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    batch_job = client.batches.create(**job_config)
    print(batch_job)
    return batch_job


def main():
    parser = argparse.ArgumentParser(
        description="Manage SiliconFlow Batch Inference jobs."
    )
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument(
        "--upload",
        type=Path,
        metavar="FILE",
        help="Upload a JSONL batch input file.",
    )
    operation.add_argument(
        "--create_job",
        type=Path,
        metavar="FILE",
        help="Create a batch job for a previously uploaded input file.",
    )
    args = parser.parse_args()

    if args.upload is not None:
        upload_batch_file(args.upload)
    elif args.create_job is not None:
        create_batch_job(args.create_job)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
