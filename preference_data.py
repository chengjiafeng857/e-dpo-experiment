from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


HH_ASSISTANT_MARKER = "\n\nAssistant:"
HH_DATASET_NAME = "Anthropic/hh-rlhf"
HH_SUBSETS = {"helpful-base", "harmless-base"}


def config_value(config: Any, key: str, default: Any = None) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def extract_anthropic_prompt(prompt_and_response: str) -> str:
    search_term_idx = prompt_and_response.rfind(HH_ASSISTANT_MARKER)
    if search_term_idx == -1:
        raise ValueError(f"Prompt and response does not contain '{HH_ASSISTANT_MARKER}'")
    return prompt_and_response[: search_term_idx + len(HH_ASSISTANT_MARKER)]


def split_hh_row(row: Mapping[str, Any]) -> dict[str, str]:
    chosen = row["chosen"]
    rejected = row["rejected"]
    prompt = extract_anthropic_prompt(chosen)
    if not rejected.startswith(prompt):
        raise ValueError("Rejected response does not share the extracted Anthropic prompt prefix")

    return {
        "prompt": prompt,
        "chosen": chosen[len(prompt) :],
        "rejected": rejected[len(prompt) :],
    }


def hh_row_within_length(
    row: Mapping[str, str],
    tokenizer: Any,
    max_length: int,
    max_prompt_length: int,
) -> bool:
    prompt_tokens = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
    chosen_tokens = tokenizer(row["chosen"], add_special_tokens=False)["input_ids"]
    rejected_tokens = tokenizer(row["rejected"], add_special_tokens=False)["input_ids"]
    return (
        len(prompt_tokens) <= max_prompt_length
        and len(prompt_tokens) + len(chosen_tokens) <= max_length
        and len(prompt_tokens) + len(rejected_tokens) <= max_length
    )


def normalize_hh_rows(
    rows: list[Mapping[str, Any]],
    tokenizer: Any,
    max_length: int,
    max_prompt_length: int,
) -> list[dict[str, str]]:
    normalized_rows = []
    for row in rows:
        normalized_row = split_hh_row(row)
        if hh_row_within_length(normalized_row, tokenizer, max_length, max_prompt_length):
            normalized_rows.append(normalized_row)
    return normalized_rows


def build_hh_dataset_load_kwargs(dataset_config: Any, split: str) -> dict[str, str]:
    dataset_name = config_value(dataset_config, "name", HH_DATASET_NAME)
    if dataset_name != HH_DATASET_NAME:
        raise ValueError(f"Unsupported HH dataset '{dataset_name}'. Expected '{HH_DATASET_NAME}'")

    data_dir = config_value(dataset_config, "data_dir")
    if data_dir not in HH_SUBSETS:
        raise ValueError(f"Unsupported HH data_dir '{data_dir}'. Expected one of: {sorted(HH_SUBSETS)}")

    return {"dataset_name": dataset_name, "data_dir": data_dir, "split": split}


def write_hh_debug_log(
    normalized_rows: list[Mapping[str, str]],
    tokenizer: Any,
    data_dir: str,
    split: str,
    max_length: int,
    max_prompt_length: int,
    total_rows: int,
    debug_dir: str | Path = "debug_log",
) -> Path:
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    log_path = debug_dir / f"hh_{data_dir}_{split}_samples.log"

    lines = [
        f"dataset: {HH_DATASET_NAME}",
        f"data_dir: {data_dir}",
        f"split: {split}",
        f"raw_rows: {total_rows}",
        f"normalized_rows: {len(normalized_rows)}",
        f"max_length: {max_length}",
        f"max_prompt_length: {max_prompt_length}",
        f"samples_written: {min(3, len(normalized_rows))}",
        "",
    ]

    for sample_index, row in enumerate(normalized_rows[:3], start=1):
        prompt_tokens = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        chosen_tokens = tokenizer(row["chosen"], add_special_tokens=False)["input_ids"]
        rejected_tokens = tokenizer(row["rejected"], add_special_tokens=False)["input_ids"]
        lines.extend(
            [
                f"=== sample_{sample_index} ===",
                f"prompt_tokens: {len(prompt_tokens)}",
                f"chosen_tokens: {len(chosen_tokens)}",
                f"rejected_tokens: {len(rejected_tokens)}",
                "[prompt]",
                row["prompt"],
                "[chosen]",
                row["chosen"],
                "[rejected]",
                row["rejected"],
                "",
            ]
        )

    log_path.write_text("\n".join(lines), encoding="utf-8")
    return log_path


def load_hh_dataset(
    dataset_config: Any,
    split: str,
    tokenizer: Any,
    max_length: int,
    max_prompt_length: int,
):
    from datasets import Dataset, load_dataset

    load_kwargs = build_hh_dataset_load_kwargs(dataset_config, split)
    rows = load_dataset(load_kwargs["dataset_name"], data_dir=load_kwargs["data_dir"], split=load_kwargs["split"])
    normalized_rows = normalize_hh_rows(rows, tokenizer, max_length, max_prompt_length)
    if not normalized_rows:
        raise ValueError(
            "No HH rows remain after DPO-style normalization and filtering for "
            f"data_dir='{load_kwargs['data_dir']}', split='{split}'"
        )
    write_hh_debug_log(
        normalized_rows=normalized_rows,
        tokenizer=tokenizer,
        data_dir=load_kwargs["data_dir"],
        split=split,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        total_rows=len(rows),
    )
    return Dataset.from_list(normalized_rows)


def select_preference_columns(dataset):
    columns = ["chosen", "rejected"]
    if "prompt" in dataset.column_names:
        columns.insert(0, "prompt")

    missing_columns = [column for column in columns if column not in dataset.column_names]
    if missing_columns:
        raise ValueError(f"Dataset is missing required preference columns: {missing_columns}")

    return dataset.select_columns(columns)


def load_preference_dataset(dataset_config: Any, split: str, tokenizer: Any, training_args: Any):
    if config_value(dataset_config, "name") == HH_DATASET_NAME and config_value(dataset_config, "data_dir") in HH_SUBSETS:
        return load_hh_dataset(
            dataset_config=dataset_config,
            split=split,
            tokenizer=tokenizer,
            max_length=training_args.max_length,
            max_prompt_length=training_args.max_prompt_length,
        )

    from datasets import load_dataset

    dataset_dict = load_dataset(config_value(dataset_config, "name"))
    return select_preference_columns(dataset_dict[split])
