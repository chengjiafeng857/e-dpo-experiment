import re
from typing import Any, Optional

from datasets import Dataset, DatasetDict, load_dataset


HH_ROLE_PATTERN = re.compile(r"(?:^|\n\n)(Human|Humans|Assistant): ")
SUPPORTED_HH_CONFIGS = {"helpful-base", "harmless-base"}
PREFERENCE_COLUMNS = ["chosen", "rejected"]


def load_preference_datasets(dataset_config: Any) -> DatasetDict:
    config_name = dataset_config.get("config_name")
    dataset = load_dataset(dataset_config.name, name=config_name)
    return normalize_preference_dataset(dataset, dataset_config.name, config_name=config_name)


def normalize_preference_dataset(
    dataset: DatasetDict, dataset_name: str, config_name: Optional[str] = None
) -> DatasetDict:
    if dataset_name.startswith("princeton-nlp/") and dataset_name.endswith("ultrafeedback"):
        return DatasetDict({split: ds.select_columns(PREFERENCE_COLUMNS) for split, ds in dataset.items()})

    if dataset_name == "Anthropic/hh-rlhf":
        if config_name not in SUPPORTED_HH_CONFIGS:
            raise ValueError(
                f"Unsupported config_name for {dataset_name}: {config_name!r}. "
                f"Expected one of {sorted(SUPPORTED_HH_CONFIGS)}."
            )
        return DatasetDict({split: _normalize_hh_split(ds) for split, ds in dataset.items()})

    raise ValueError(f"Unsupported dataset for preference training: {dataset_name!r}")


def parse_hh_transcript(transcript: str) -> list[dict[str, str]]:
    parts = HH_ROLE_PATTERN.split(transcript)
    if parts[0].strip():
        return []

    messages = []
    idx = 1
    while idx < len(parts):
        role = parts[idx]
        content = parts[idx + 1].strip() if idx + 1 < len(parts) else ""
        mapped_role = "user" if role in {"Human", "Humans"} else "assistant"
        messages.append({"role": mapped_role, "content": content})
        idx += 2

    return messages


def is_valid_preference_pair(chosen: list[dict[str, str]], rejected: list[dict[str, str]]) -> bool:
    if not chosen or not rejected:
        return False

    if not _is_valid_conversation(chosen) or not _is_valid_conversation(rejected):
        return False

    common_prefix_len = 0
    for chosen_message, rejected_message in zip(chosen, rejected):
        if chosen_message != rejected_message:
            break
        common_prefix_len += 1

    return common_prefix_len > 0


def _normalize_hh_split(dataset: Dataset) -> Dataset:
    normalized = dataset.map(_normalize_hh_example, desc="Normalizing HH-RLHF transcripts")
    normalized = normalized.filter(lambda example: example["valid"], desc="Filtering invalid HH-RLHF rows")
    return normalized.select_columns(PREFERENCE_COLUMNS)


def _normalize_hh_example(example: dict[str, Any]) -> dict[str, Any]:
    chosen = parse_hh_transcript(example["chosen"])
    rejected = parse_hh_transcript(example["rejected"])
    return {
        "chosen": chosen,
        "rejected": rejected,
        "valid": is_valid_preference_pair(chosen, rejected),
    }


def _is_valid_conversation(messages: list[dict[str, str]]) -> bool:
    if not messages:
        return False

    if messages[0]["role"] != "user" or messages[-1]["role"] != "assistant":
        return False

    previous_role = None
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content:
            return False
        if role == previous_role:
            return False
        previous_role = role

    return True
