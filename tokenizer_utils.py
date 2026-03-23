from __future__ import annotations

import logging
from typing import Any


LOGGER = logging.getLogger(__name__)


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


def load_tokenizer(config: Any):
    from transformers import AutoTokenizer

    candidates = resolve_tokenizer_candidates(config)
    errors: list[str] = []

    for candidate in candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate)
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
