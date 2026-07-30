from pathlib import Path

from openai import OpenAI
import yaml


API_CONFIG_PATH = Path(__file__).resolve().parent / "api_config.yaml"


def load_siliconflow_config(api_config_path=API_CONFIG_PATH):
    """Load the SiliconFlow API key and base URL from the local YAML config."""
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


api_key, base_url = load_siliconflow_config()
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

batch_input_file = client.files.create(
    file=open("data/prompt_for_syn_batch_infer.jsonl", "rb"),
    purpose="batch"
)
print(batch_input_file)
# ID returned after upload
file_id = batch_input_file.data['id']
print(file_id)
