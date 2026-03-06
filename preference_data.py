from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping


HH_HUMAN_MARKER = "\n\nHuman:"
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


def strip_anthropic_turn_prefix(text: str) -> str:
    return text[1:] if text.startswith(" ") else text


def hh_prompt_to_messages(prompt: str) -> list[dict[str, str]]:
    if not prompt.endswith(HH_ASSISTANT_MARKER):
        raise ValueError("HH prompt must end with the final assistant marker")

    parts = re.split(r"\n\n(Human|Assistant):", prompt)
    if not parts or parts[0] != "":
        raise ValueError("Unsupported HH prompt format")

    messages = []
    turns = parts[1:]
    if len(turns) % 2 != 0:
        raise ValueError("Malformed HH prompt turns")

    for idx in range(0, len(turns), 2):
        role = turns[idx]
        content = strip_anthropic_turn_prefix(turns[idx + 1])
        is_last_turn = idx == len(turns) - 2
        if is_last_turn:
            if role != "Assistant" or content:
                raise ValueError("HH prompt must terminate with an empty assistant turn")
            continue
        messages.append(
            {
                "role": "user" if role == "Human" else "assistant",
                "content": content,
            }
        )

    if not messages or messages[-1]["role"] != "user":
        raise ValueError("HH prompt must end on a user turn before generation")

    return messages


def hh_row_to_conversational(row: Mapping[str, str]) -> dict[str, list[dict[str, str]]]:
    return {
        "prompt": hh_prompt_to_messages(row["prompt"]),
        "chosen": [{"role": "assistant", "content": strip_anthropic_turn_prefix(row["chosen"])}],
        "rejected": [{"role": "assistant", "content": strip_anthropic_turn_prefix(row["rejected"])}],
    }


def render_preference_row(row: Mapping[str, Any], tokenizer: Any) -> dict[str, str]:
    if isinstance(row["prompt"], str):
        return {
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
        }

    prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
    prompt_chosen = tokenizer.apply_chat_template(row["prompt"] + row["chosen"], tokenize=False)
    prompt_rejected = tokenizer.apply_chat_template(row["prompt"] + row["rejected"], tokenize=False)

    if not prompt_chosen.startswith(prompt) or not prompt_rejected.startswith(prompt):
        raise ValueError("Tokenizer chat template is incompatible with HH conversational formatting")

    return {
        "prompt": prompt,
        "chosen": prompt_chosen[len(prompt) :],
        "rejected": prompt_rejected[len(prompt) :],
    }


def hh_row_within_length(
    row: Mapping[str, Any],
    tokenizer: Any,
    max_length: int,
    max_prompt_length: int,
) -> bool:
    rendered_row = render_preference_row(row, tokenizer)
    prompt_tokens = tokenizer(rendered_row["prompt"], add_special_tokens=False)["input_ids"]
    chosen_tokens = tokenizer(rendered_row["chosen"], add_special_tokens=False)["input_ids"]
    rejected_tokens = tokenizer(rendered_row["rejected"], add_special_tokens=False)["input_ids"]
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
    apply_chat_template: bool = False,
) -> list[dict[str, Any]]:
    normalized_rows = []
    for row in rows:
        normalized_row = split_hh_row(row)
        if apply_chat_template:
            normalized_row = hh_row_to_conversational(normalized_row)
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
    normalized_rows: list[Mapping[str, Any]],
    tokenizer: Any,
    data_dir: str,
    split: str,
    max_length: int,
    max_prompt_length: int,
    total_rows: int,
    apply_chat_template: bool = False,
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
        f"apply_chat_template: {apply_chat_template}",
        f"samples_written: {min(3, len(normalized_rows))}",
        "",
    ]

    for sample_index, row in enumerate(normalized_rows[:3], start=1):
        rendered_row = render_preference_row(row, tokenizer)
        prompt_tokens = tokenizer(rendered_row["prompt"], add_special_tokens=False)["input_ids"]
        chosen_tokens = tokenizer(rendered_row["chosen"], add_special_tokens=False)["input_ids"]
        rejected_tokens = tokenizer(rendered_row["rejected"], add_special_tokens=False)["input_ids"]
        lines.extend(
            [
                f"=== sample_{sample_index} ===",
                f"stored_prompt_type: {type(row['prompt']).__name__}",
                f"prompt_tokens: {len(prompt_tokens)}",
                f"chosen_tokens: {len(chosen_tokens)}",
                f"rejected_tokens: {len(rejected_tokens)}",
                "[prompt]",
                rendered_row["prompt"],
                "[chosen]",
                rendered_row["chosen"],
                "[rejected]",
                rendered_row["rejected"],
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

    apply_chat_template = bool(config_value(dataset_config, "apply_chat_template", False))
    load_kwargs = build_hh_dataset_load_kwargs(dataset_config, split)
    rows = load_dataset(load_kwargs["dataset_name"], data_dir=load_kwargs["data_dir"], split=load_kwargs["split"])
    normalized_rows = normalize_hh_rows(
        rows,
        tokenizer,
        max_length,
        max_prompt_length,
        apply_chat_template=apply_chat_template,
    )
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
        apply_chat_template=apply_chat_template,
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
