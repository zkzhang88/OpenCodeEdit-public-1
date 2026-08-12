from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .io import InferenceError, load_yaml


DEFAULTS: dict[str, Any] = {
    "sampling": {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 2048,
    },
    "num_completion": 1,
    "max_batch_attempts": 3,
    "executors": {
        "api": {"request_timeout": 600, "request_retries": 5},
        "siliconflow-batch": {
            "api_key_field": "SILICONFLOW_API_KEY",
            "base_url_field": "SILICONFLOW_BASE_URL",
            "completion_window": "24h",
            "endpoint": "/v1/chat/completions",
            "max_requests_per_file": 6000,
            "poll_interval": 60,
        },
        "llm-infer": {
            "conda_executable": "conda",
            "conda_environment": "llm_infer",
            "command": "batch-infer",
            "auto_serve": True,
            "concurrency": 8,
            "timeout": 600,
            "max_retries": 3,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
            "models_dir": "~/models",
        },
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_config(
    config_path: str | Path,
    executor: str,
    model_profile: str,
    cli_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    loaded = load_yaml(config_path)
    merged = deep_merge(DEFAULTS, loaded.get("defaults", {}))
    merged["executors"] = deep_merge(
        merged.get("executors", {}), loaded.get("executors", {})
    )

    profiles = loaded.get("models", {})
    if not isinstance(profiles, dict) or model_profile not in profiles:
        raise InferenceError(f"Unknown model profile: {model_profile}")
    profile = profiles[model_profile]
    if not isinstance(profile, dict):
        raise InferenceError(f"Model profile {model_profile} must be a mapping")
    executor_profile = profile.get(executor)
    if not isinstance(executor_profile, dict):
        raise InferenceError(
            f"Model profile {model_profile} does not support executor {executor}"
        )

    executor_defaults = merged.get("executors", {}).get(executor)
    if not isinstance(executor_defaults, dict):
        raise InferenceError(f"Missing executor configuration: {executor}")

    configured_api_path = Path(
        loaded.get("api_config_path", config_path.with_name("api_config.yaml"))
    ).expanduser()
    if not configured_api_path.is_absolute():
        configured_api_path = config_path.parent / configured_api_path

    resolved = {
        "config_path": str(config_path),
        "api_config_path": str(configured_api_path.resolve()),
        "executor": executor,
        "model_profile": model_profile,
        "num_completion": merged["num_completion"],
        "max_batch_attempts": merged["max_batch_attempts"],
        "sampling": deepcopy(merged["sampling"]),
        "executor_config": deep_merge(executor_defaults, executor_profile),
    }
    overrides = {key: value for key, value in (cli_overrides or {}).items() if value is not None}
    for key in ("num_completion", "max_batch_attempts"):
        if key in overrides:
            resolved[key] = overrides.pop(key)
    sampling = resolved["sampling"]
    for key in ("temperature", "top_p", "max_tokens"):
        if key in overrides:
            sampling[key] = overrides.pop(key)
    resolved["executor_config"] = deep_merge(resolved["executor_config"], overrides)
    validate_resolved_config(resolved)
    return resolved


def validate_resolved_config(config: dict[str, Any]) -> None:
    if config["num_completion"] < 1:
        raise InferenceError("num_completion must be at least 1")
    if config["max_batch_attempts"] < 1:
        raise InferenceError("max_batch_attempts must be at least 1")
    sampling = config["sampling"]
    if sampling["max_tokens"] < 1:
        raise InferenceError("max_tokens must be at least 1")
    executor_config = config["executor_config"]
    model = executor_config.get("model")
    if not isinstance(model, str) or not model.strip():
        raise InferenceError(
            f"Executor {config['executor']} requires a non-empty model setting"
        )
    if config["executor"] == "api":
        for field in ("api_key_field", "base_url_field"):
            if not executor_config.get(field):
                raise InferenceError(f"API executor requires {field}")
    if config["executor"] == "llm-infer":
        if not executor_config.get("auto_serve", True) and not executor_config.get(
            "base_url"
        ):
            raise InferenceError("llm-infer without auto_serve requires base_url")
        visible_devices = executor_config.get("visible_devices")
        if visible_devices is not None:
            device_count = len(
                [item for item in str(visible_devices).split(",") if item.strip()]
            )
            if device_count != int(executor_config.get("tensor_parallel_size", 1)):
                raise InferenceError(
                    "llm-infer tensor_parallel_size must match visible_devices count"
                )


def validated_config_snapshot(
    config: object, executor: str, model_profile: str
) -> dict[str, Any]:
    """Return an isolated, validated copy of a previously resolved config."""

    if not isinstance(config, dict):
        raise InferenceError("Resolved inference config snapshot must be a mapping")
    snapshot = deepcopy(config)
    if snapshot.get("executor") != executor:
        raise InferenceError(
            "Resolved inference config executor does not match the requested executor"
        )
    if snapshot.get("model_profile") != model_profile:
        raise InferenceError(
            "Resolved inference config model profile does not match the requested model"
        )
    try:
        validate_resolved_config(snapshot)
    except InferenceError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise InferenceError(
            f"Invalid resolved inference config snapshot: {error}"
        ) from error
    return snapshot


def load_credentials(config: dict[str, Any]) -> tuple[str, str]:
    executor_config = config["executor_config"]
    credentials = load_yaml(config["api_config_path"])
    key_field = executor_config["api_key_field"]
    url_field = executor_config["base_url_field"]
    missing = [field for field in (key_field, url_field) if not credentials.get(field)]
    if missing:
        raise InferenceError(
            "Missing required fields in API config: " + ", ".join(missing)
        )
    return str(credentials[key_field]), str(credentials[url_field])
