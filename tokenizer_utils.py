from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from typing import Any


LOGGER = logging.getLogger(__name__)

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "vocab.txt",
    "sentencepiece.bpe.model",
    "spiece.model",
    "config.json",
    "generation_config.json",
)


def config_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _append_candidate(candidates: list[str], candidate: str | None) -> None:
    if candidate and candidate not in candidates:
        candidates.append(candidate)


def resolve_tokenizer_candidates(config: Any) -> list[str]:
    tokenizer_config = config_value(config, "tokenizer", {})
    model_config = config_value(config, "model", {})

    candidates: list[str] = []
    _append_candidate(
        candidates,
        config_value(tokenizer_config, "name_or_path", config_value(tokenizer_config, "original_name_or_path")),
    )
    _append_candidate(candidates, config_value(tokenizer_config, "fallback_name_or_path"))
    _append_candidate(candidates, config_value(model_config, "pretrained_model_name_or_path"))

    if not candidates:
        raise ValueError("Could not resolve any tokenizer candidates from the config")

    return candidates


def maybe_apply_chat_template(tokenizer: Any, config: Any) -> bool:
    if getattr(tokenizer, "chat_template", None):
        return False

    dataset_config = config_value(config, "dataset", {})
    if not config_value(dataset_config, "apply_chat_template", False):
        return False

    tokenizer_config = config_value(config, "tokenizer", {})
    explicit_template = config_value(tokenizer_config, "chat_template")
    if explicit_template:
        tokenizer.chat_template = explicit_template
        return True

    return False


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _copy_tokenizer_files(candidate: str, destination_dir: str) -> list[str]:
    from transformers.utils.hub import cached_file

    copied_files: list[str] = []
    for filename in TOKENIZER_FILES:
        try:
            source_path = cached_file(candidate, filename)
        except Exception:  # pragma: no cover
            continue
        shutil.copy2(source_path, os.path.join(destination_dir, filename))
        copied_files.append(filename)
    return copied_files


def _load_tokenizer_with_repaired_config(candidate: str):
    from transformers import AutoTokenizer
    from transformers.utils.hub import cached_file

    tokenizer_config_path = cached_file(candidate, "tokenizer_config.json")
    tokenizer_config = _read_json(tokenizer_config_path)
    if "tokenizer_class" not in tokenizer_config:
        raise ValueError(f"Tokenizer '{candidate}' does not declare tokenizer_class in tokenizer_config.json.")

    with tempfile.TemporaryDirectory(prefix="tokenizer-repair-") as tempdir:
        copied_files = _copy_tokenizer_files(candidate, tempdir)
        if "tokenizer_config.json" not in copied_files:
            raise ValueError(f"Tokenizer '{candidate}' did not provide tokenizer_config.json for repair.")

        tokenizer_config.pop("tokenizer_class", None)
        with open(os.path.join(tempdir, "tokenizer_config.json"), "w", encoding="utf-8") as handle:
            json.dump(tokenizer_config, handle, indent=2, sort_keys=True)
            handle.write("\n")

        return AutoTokenizer.from_pretrained(tempdir)


def _load_tokenizers_backend_tokenizer(candidate: str):
    from transformers import PreTrainedTokenizerFast
    from transformers.utils.hub import cached_file

    tokenizer_config_path = cached_file(candidate, "tokenizer_config.json")
    tokenizer_config = _read_json(tokenizer_config_path)
    tokenizer_class = tokenizer_config.get("tokenizer_class")
    backend = tokenizer_config.get("backend")
    if tokenizer_class != "TokenizersBackend" and backend != "tokenizers":
        raise ValueError(
            f"Tokenizer '{candidate}' is not using the tokenizers backend fallback "
            f"(tokenizer_class={tokenizer_class!r}, backend={backend!r})."
        )

    tokenizer_file = cached_file(candidate, "tokenizer.json")
    init_kwargs = {
        key: tokenizer_config[key]
        for key in (
            "bos_token",
            "eos_token",
            "unk_token",
            "sep_token",
            "pad_token",
            "cls_token",
            "mask_token",
            "model_max_length",
        )
        if key in tokenizer_config
    }
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, **init_kwargs)

    for attribute in ("clean_up_tokenization_spaces", "padding_side", "truncation_side", "model_input_names"):
        if attribute in tokenizer_config:
            setattr(tokenizer, attribute, tokenizer_config[attribute])

    chat_template = tokenizer_config.get("chat_template")
    if not chat_template:
        try:
            chat_template_path = cached_file(candidate, "chat_template.jinja")
        except Exception:  # pragma: no cover
            chat_template_path = None
        if chat_template_path:
            chat_template = _read_text(chat_template_path)
    if chat_template:
        tokenizer.chat_template = chat_template

    return tokenizer


def load_tokenizer(config: Any):
    from transformers import AutoTokenizer

    candidates = resolve_tokenizer_candidates(config)
    errors: list[str] = []

    for candidate in candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate)
        except Exception as auto_exc:  # pragma: no cover
            try:
                tokenizer = _load_tokenizer_with_repaired_config(candidate)
                LOGGER.warning(
                    "Tokenizer '%s' could not be loaded via AutoTokenizer; "
                    "repaired tokenizer_config.json locally and retried AutoTokenizer (%s: %s).",
                    candidate,
                    type(auto_exc).__name__,
                    auto_exc,
                )
            except Exception as repaired_exc:
                try:
                    tokenizer = _load_tokenizers_backend_tokenizer(candidate)
                    LOGGER.warning(
                        "Tokenizer '%s' could not be loaded via AutoTokenizer or repaired config; "
                        "fell back to PreTrainedTokenizerFast from tokenizer.json (%s: %s).",
                        candidate,
                        type(auto_exc).__name__,
                        auto_exc,
                    )
                except Exception as fallback_exc:
                    errors.append(f"{candidate}: {type(auto_exc).__name__}: {auto_exc}")
                    errors.append(
                        f"{candidate} [repaired-config fallback]: "
                        f"{type(repaired_exc).__name__}: {repaired_exc}"
                    )
                    errors.append(
                        f"{candidate} [tokenizers-backend fallback]: "
                        f"{type(fallback_exc).__name__}: {fallback_exc}"
                    )
                    continue

        try:
            template_applied = maybe_apply_chat_template(tokenizer, config)
            if template_applied:
                LOGGER.warning(
                    "Tokenizer '%s' did not define a chat template; applied the explicitly configured template.",
                    candidate,
                )
            elif (
                config_value(config_value(config, "dataset", {}), "apply_chat_template", False)
                and not getattr(tokenizer, "chat_template", None)
            ):
                raise ValueError(
                    f"Tokenizer '{candidate}' does not provide a chat template. "
                    "Disable dataset.apply_chat_template or configure tokenizer.chat_template explicitly."
                )
            return tokenizer
        except Exception as exc:  # pragma: no cover
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")

    candidate_list = ", ".join(candidates)
    error_details = "\n".join(errors)
    raise RuntimeError(
        "Failed to load a tokenizer from any configured source. "
        f"Tried, in order: {candidate_list}\n{error_details}"
    )
