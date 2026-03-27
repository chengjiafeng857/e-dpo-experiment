from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

from chat_templates import (
    ensure_tokenizer_chat_template,
    get_assistant_generation_suffix,
    parse_hh_to_messages,
    strip_one_leading_newline,
)


HH_HUMAN_MARKER = "\n\nHuman:"
HH_ASSISTANT_MARKER = "\n\nAssistant:"
HH_DATASET_NAME = "Anthropic/hh-rlhf"
HH_SUBSETS = {"helpful-base", "harmless-base"}
SAMPLE_PREVIEW_CHARS = 512


def config_value(config: Any, key: str, default: Any = None) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _is_primary_process() -> bool:
    return os.environ.get("RANK", "0") == "0"


def _truncate_preview(text: Any, limit: int = SAMPLE_PREVIEW_CHARS) -> str:
    normalized = _normalize_text(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "\n...<truncated>..."


def print_preprocessed_sample(dataset: Any, *, split: str) -> None:
    if not _is_primary_process():
        return
    if len(dataset) == 0:
        return

    row = dataset[0]
    print(f"\n[preprocessed-sample] split={split} columns={list(dataset.column_names)}")
    if "prompt" in row:
        print("[prompt]")
        print(_truncate_preview(row["prompt"]))
    if "chosen" in row:
        print("[chosen]")
        print(_truncate_preview(row["chosen"]))
    if "rejected" in row:
        print("[rejected]")
        print(_truncate_preview(row["rejected"]))
    print()


def _normalize_text(text: Any) -> str:
    return str(text).replace("\r\n", "\n").replace("\r", "\n")


def dataset_uses_chat_template(dataset_config: Any) -> bool:
    return bool(
        config_value(dataset_config, "apply_chat_template", config_value(dataset_config, "chat_template", False))
    )


def extract_anthropic_prompt(prompt_and_response: str) -> str:
    search_term_idx = prompt_and_response.rfind(HH_ASSISTANT_MARKER)
    if search_term_idx == -1:
        raise ValueError(f"Prompt and response does not contain '{HH_ASSISTANT_MARKER}'")
    return prompt_and_response[: search_term_idx + len(HH_ASSISTANT_MARKER)]


def extract_shared_anthropic_prompt(chosen: str, rejected: str) -> str:
    common_prefix_length = 0
    for chosen_char, rejected_char in zip(chosen, rejected):
        if chosen_char != rejected_char:
            break
        common_prefix_length += 1

    if common_prefix_length == 0:
        raise ValueError("Chosen and rejected responses do not share a common prompt prefix")

    return extract_anthropic_prompt(chosen[:common_prefix_length])


def split_hh_row(row: Mapping[str, Any]) -> dict[str, str]:
    chosen = row["chosen"]
    rejected = row["rejected"]
    prompt = extract_shared_anthropic_prompt(chosen, rejected)
    if not rejected.startswith(prompt):
        raise ValueError("Rejected response does not share the extracted Anthropic prompt prefix")

    return {
        "prompt": prompt,
        "chosen": chosen[len(prompt) :],
        "rejected": rejected[len(prompt) :],
    }


def split_prompt_and_response(input_text: str) -> tuple[str, str]:
    input_text = _normalize_text(input_text)
    index = input_text.rfind(HH_ASSISTANT_MARKER)
    if index < 0:
        raise ValueError(f"Prompt and response does not contain '{HH_ASSISTANT_MARKER}'")
    prompt = input_text[: index + len(HH_ASSISTANT_MARKER)]
    response = strip_one_leading_newline(input_text[index + len(HH_ASSISTANT_MARKER) :])
    return prompt, response


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


def convert_hh_row(row: Mapping[str, Any]) -> dict[str, str]:
    chosen_prompt, chosen_response = split_prompt_and_response(row["chosen"])
    rejected_text = _normalize_text(row["rejected"])
    if not rejected_text.startswith(chosen_prompt):
        raise ValueError("Rejected response does not share the extracted Anthropic prompt prefix")

    rejected_response = strip_one_leading_newline(rejected_text[len(chosen_prompt) :])

    if len(chosen_prompt.strip()) == 0:
        raise ValueError("Chosen prompt is empty after HH normalization")
    if len(chosen_response.strip()) == 0 or len(rejected_response.strip()) == 0:
        raise ValueError("Chosen or rejected response is empty after HH normalization")

    return {
        "prompt": chosen_prompt,
        "chosen": chosen_response,
        "rejected": rejected_response,
    }


def convert_to_triples(chosen_text: str, rejected_text: str) -> dict[str, str] | None:
    try:
        return convert_hh_row({"chosen": chosen_text, "rejected": rejected_text})
    except ValueError:
        return None


def build_HH_dataset(ds):
    from datasets import Dataset

    hh_ds_raw = []
    for row in ds:
        output = convert_to_triples(
            chosen_text=row["chosen"],
            rejected_text=row["rejected"],
        )
        if output is not None:
            hh_ds_raw.append(output)
    return Dataset.from_list(hh_ds_raw)


def _ensure_generation_prompt(prompt_text: str, *, assistant_generation_suffix: str | None) -> str:
    if not assistant_generation_suffix:
        return prompt_text
    trimmed = prompt_text.rstrip()
    if trimmed.endswith(assistant_generation_suffix.rstrip()):
        return prompt_text
    return f"{prompt_text}{assistant_generation_suffix}"


def _render_response_with_chat_template(
    messages: list[dict[str, str]],
    response: str,
    *,
    tokenizer: Any,
    prompt_text: str,
) -> str:
    response = _normalize_text(response).strip()
    if not response:
        raise ValueError("HH response is empty after chat-template normalization")

    full_messages = messages + [{"role": "assistant", "content": response}]
    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    if full_text.startswith(prompt_text):
        rendered = full_text[len(prompt_text) :]
    else:
        rendered = response
    rendered = rendered.strip()
    if not rendered:
        raise ValueError("Rendered HH response is empty after chat-template normalization")
    return rendered


def apply_chat_template_to_hh_row(
    row: Mapping[str, str],
    tokenizer: Any,
    *,
    model_name: str = "",
    chat_template_name: str | None = None,
) -> dict[str, str]:
    resolved_template_name = ensure_tokenizer_chat_template(
        tokenizer,
        model_name=model_name or getattr(tokenizer, "name_or_path", ""),
        configured_name=chat_template_name,
    )
    assistant_generation_suffix = None
    if resolved_template_name is not None:
        assistant_generation_suffix = get_assistant_generation_suffix(resolved_template_name)

    prompt_text = _normalize_text(row["prompt"]).strip()
    if not prompt_text:
        raise ValueError("HH prompt is empty after normalization")

    messages = parse_hh_to_messages(prompt_text)
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("HH prompt must end on a user turn before generation")

    prompt_rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not prompt_rendered:
        raise ValueError("Tokenizer returned an empty chat-templated HH prompt")
    prompt_rendered = _ensure_generation_prompt(
        prompt_rendered,
        assistant_generation_suffix=assistant_generation_suffix,
    )

    chosen_rendered = _render_response_with_chat_template(
        messages,
        row["chosen"],
        tokenizer=tokenizer,
        prompt_text=prompt_rendered,
    )
    rejected_rendered = _render_response_with_chat_template(
        messages,
        row["rejected"],
        tokenizer=tokenizer,
        prompt_text=prompt_rendered,
    )

    return {
        "prompt": prompt_rendered,
        "chosen": chosen_rendered,
        "rejected": rejected_rendered,
    }


def apply_chat_template_to_dataset(
    ds,
    tokenizer: Any,
    *,
    model_name: str = "",
    chat_template_name: str | None = None,
):
    from datasets import Dataset

    resolved_template_name = ensure_tokenizer_chat_template(
        tokenizer,
        model_name=model_name or getattr(tokenizer, "name_or_path", ""),
        configured_name=chat_template_name,
    )
    assistant_generation_suffix = None
    if resolved_template_name is not None:
        assistant_generation_suffix = get_assistant_generation_suffix(resolved_template_name)

    rows: list[dict[str, str]] = []
    for row in ds:
        prompt_text = _normalize_text(row.get("prompt", "")).strip()
        chosen_text = row.get("chosen", "")
        rejected_text = row.get("rejected", "")
        if not prompt_text:
            continue

        messages = parse_hh_to_messages(prompt_text)
        if not messages or messages[-1]["role"] != "user":
            continue

        prompt_rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if not prompt_rendered:
            continue
        prompt_rendered = _ensure_generation_prompt(
            prompt_rendered,
            assistant_generation_suffix=assistant_generation_suffix,
        )

        try:
            chosen_rendered = _render_response_with_chat_template(
                messages,
                str(chosen_text),
                tokenizer=tokenizer,
                prompt_text=prompt_rendered,
            )
            rejected_rendered = _render_response_with_chat_template(
                messages,
                str(rejected_text),
                tokenizer=tokenizer,
                prompt_text=prompt_rendered,
            )
        except ValueError:
            continue

        rows.append(
            {
                "prompt": prompt_rendered,
                "chosen": chosen_rendered,
                "rejected": rejected_rendered,
            }
        )
    return Dataset.from_list(rows)


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
def normalize_hh_rows(
    rows: list[Mapping[str, Any]],
    tokenizer: Any,
    max_length: int,
    max_prompt_length: int | None,
    apply_chat_template: bool = False,
    model_name: str = "",
    chat_template_name: str | None = None,
    return_stats: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, int]]:
    normalized_rows = []
    skipped_invalid_rows = 0
    for row in rows:
        try:
            normalized_row = convert_hh_row(row)
            if apply_chat_template:
                normalized_row = apply_chat_template_to_hh_row(
                    normalized_row,
                    tokenizer,
                    model_name=model_name,
                    chat_template_name=chat_template_name,
                )
        except ValueError:
            skipped_invalid_rows += 1
            continue
        normalized_rows.append(normalized_row)
    if return_stats:
        return normalized_rows, {"skipped_invalid_rows": skipped_invalid_rows}
    return normalized_rows


def build_hh_dataset_load_kwargs(dataset_config: Any, split: str) -> dict[str, str]:
    dataset_name = config_value(dataset_config, "name", config_value(dataset_config, "dataset_name", HH_DATASET_NAME))
    if dataset_name != HH_DATASET_NAME:
        raise ValueError(f"Unsupported HH dataset '{dataset_name}'. Expected '{HH_DATASET_NAME}'")

    data_dir = config_value(dataset_config, "data_dir", config_value(dataset_config, "config_name"))
    if data_dir not in HH_SUBSETS:
        raise ValueError(f"Unsupported HH data_dir '{data_dir}'. Expected one of: {sorted(HH_SUBSETS)}")

    return {"dataset_name": dataset_name, "data_dir": data_dir, "split": split}


def write_hh_debug_log(
    normalized_rows: list[Mapping[str, Any]],
    tokenizer: Any,
    data_dir: str,
    split: str,
    max_length: int,
    max_prompt_length: int | None,
    total_rows: int,
    skipped_invalid_rows: int = 0,
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
        f"skipped_invalid_rows: {skipped_invalid_rows}",
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
    max_prompt_length: int | None,
    model_name: str = "",
):
    from datasets import load_dataset

    apply_chat_template = dataset_uses_chat_template(dataset_config)
    load_kwargs = build_hh_dataset_load_kwargs(dataset_config, split)
    rows = load_dataset(load_kwargs["dataset_name"], data_dir=load_kwargs["data_dir"], split=load_kwargs["split"])
    hh_dataset = build_HH_dataset(rows)
    total_rows = len(rows)
    skipped_invalid_rows = total_rows - len(hh_dataset)
    data_dir = load_kwargs["data_dir"]

    if apply_chat_template:
        hh_dataset = apply_chat_template_to_dataset(
            hh_dataset,
            tokenizer,
            model_name=model_name,
            chat_template_name=config_value(dataset_config, "chat_template_name"),
        )

    normalized_rows = list(hh_dataset)
    if not normalized_rows:
        raise ValueError(
            "No HH rows remain after DPO-style normalization for "
            f"data_dir='{data_dir}', split='{split}'. "
            f"Skipped invalid rows: {skipped_invalid_rows}"
        )
    write_hh_debug_log(
        normalized_rows=normalized_rows,
        tokenizer=tokenizer,
        data_dir=data_dir,
        split=split,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        total_rows=total_rows,
        skipped_invalid_rows=skipped_invalid_rows,
        apply_chat_template=apply_chat_template,
    )
    return hh_dataset


def select_preference_columns(dataset):
    columns = ["chosen", "rejected"]
    if "prompt" in dataset.column_names:
        columns.insert(0, "prompt")

    missing_columns = [column for column in columns if column not in dataset.column_names]
    if missing_columns:
        raise ValueError(f"Dataset is missing required preference columns: {missing_columns}")

    return dataset.select_columns(columns)


def load_preference_dataset(
    dataset_config: Any,
    split: str,
    tokenizer: Any,
    training_args: Any,
    model_name: str = "",
):
    dataset_name = config_value(dataset_config, "name", config_value(dataset_config, "dataset_name"))
    data_dir = config_value(dataset_config, "data_dir", config_value(dataset_config, "config_name"))
    if dataset_name == HH_DATASET_NAME and data_dir in HH_SUBSETS:
        dataset = load_hh_dataset(
            dataset_config=dataset_config,
            split=split,
            tokenizer=tokenizer,
            max_length=training_args.max_length,
            max_prompt_length=training_args.max_prompt_length,
            model_name=model_name or getattr(tokenizer, "name_or_path", ""),
        )
        print_preprocessed_sample(dataset, split=split)
        return dataset

    from datasets import load_dataset

    load_kwargs = {"path": dataset_name}
    config_name = config_value(dataset_config, "config_name")
    if config_name is not None:
        load_kwargs["name"] = config_name
    dataset_dict = load_dataset(**load_kwargs)
    dataset = select_preference_columns(dataset_dict[split])
    print_preprocessed_sample(dataset, split=split)
    return dataset
