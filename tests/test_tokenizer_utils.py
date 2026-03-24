import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from tokenizer_utils import load_tokenizer, maybe_apply_chat_template, resolve_tokenizer_candidates


class FakeTokenizer:
    def __init__(self, chat_template=None):
        self.chat_template = chat_template


class TokenizerUtilsTests(unittest.TestCase):
    def test_resolve_tokenizer_candidates_prioritizes_explicit_original_tokenizer(self):
        config = {
            "model": {"pretrained_model_name_or_path": "W-61/hh-harmless-base-llama3-8b-sft"},
            "tokenizer": {"name_or_path": "meta-llama/Meta-Llama-3-8B"},
        }

        self.assertEqual(
            resolve_tokenizer_candidates(config),
            [
                "meta-llama/Meta-Llama-3-8B",
                "W-61/hh-harmless-base-llama3-8b-sft",
            ],
        )

    def test_resolve_tokenizer_candidates_keeps_explicit_fallback_before_checkpoint(self):
        config = {
            "model": {"pretrained_model_name_or_path": "checkpoint"},
            "tokenizer": {
                "name_or_path": "original-tokenizer",
                "fallback_name_or_path": "secondary-tokenizer",
            },
        }

        self.assertEqual(
            resolve_tokenizer_candidates(config),
            ["original-tokenizer", "secondary-tokenizer", "checkpoint"],
        )

    def test_maybe_apply_chat_template_does_not_infer_template(self):
        tokenizer = FakeTokenizer()
        config = {
            "dataset": {"apply_chat_template": True},
            "model": {"pretrained_model_name_or_path": "meta-llama/Meta-Llama-3-8B"},
        }

        applied = maybe_apply_chat_template(tokenizer, config)

        self.assertFalse(applied)
        self.assertIsNone(tokenizer.chat_template)

    def test_maybe_apply_chat_template_uses_explicit_template(self):
        tokenizer = FakeTokenizer()
        config = {
            "dataset": {"apply_chat_template": True},
            "tokenizer": {"chat_template": "configured-template"},
        }

        applied = maybe_apply_chat_template(tokenizer, config)

        self.assertTrue(applied)
        self.assertEqual(tokenizer.chat_template, "configured-template")

    def test_load_tokenizer_repairs_invalid_tokenizer_class_via_autotokenizer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "tokenizer_config.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "backend": "tokenizers",
                        "tokenizer_class": "BrokenTokenizerFast",
                        "bos_token": "<s>",
                        "eos_token": "</s>",
                    },
                    handle,
                )
            with open(os.path.join(tmpdir, "tokenizer.json"), "w", encoding="utf-8") as handle:
                handle.write("{}")
            with open(os.path.join(tmpdir, "config.json"), "w", encoding="utf-8") as handle:
                json.dump({"model_type": "llama"}, handle)
            with open(os.path.join(tmpdir, "chat_template.jinja"), "w", encoding="utf-8") as handle:
                handle.write("{{ messages }}")

            class FakeAutoTokenizer:
                @staticmethod
                def from_pretrained(candidate):
                    if candidate == "checkpoint":
                        raise ValueError("Tokenizer class BrokenTokenizerFast does not exist")

                    repaired_config = os.path.join(candidate, "tokenizer_config.json")
                    with open(repaired_config, encoding="utf-8") as handle:
                        repaired_data = json.load(handle)

                    if "tokenizer_class" in repaired_data:
                        raise AssertionError("tokenizer_class should be removed during repair")

                    repaired_tokenizer = FakeTokenizer(chat_template="{{ messages }}")
                    repaired_tokenizer.name_or_path = candidate
                    return repaired_tokenizer

            def fake_cached_file(_candidate, filename):
                path = os.path.join(tmpdir, filename)
                if not os.path.exists(path):
                    raise OSError(f"missing {filename}")
                return path

            fake_transformers = types.ModuleType("transformers")
            fake_transformers.AutoTokenizer = FakeAutoTokenizer
            fake_transformers_utils = types.ModuleType("transformers.utils")
            fake_transformers_hub = types.ModuleType("transformers.utils.hub")
            fake_transformers_hub.cached_file = fake_cached_file

            with patch.dict(
                sys.modules,
                {
                    "transformers": fake_transformers,
                    "transformers.utils": fake_transformers_utils,
                    "transformers.utils.hub": fake_transformers_hub,
                },
            ):
                tokenizer = load_tokenizer({"model": {"pretrained_model_name_or_path": "checkpoint"}})

        self.assertIsInstance(tokenizer, FakeTokenizer)
        self.assertEqual(tokenizer.chat_template, "{{ messages }}")

    def test_load_tokenizer_falls_back_to_tokenizers_backend_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer_config_path = os.path.join(tmpdir, "tokenizer_config.json")
            tokenizer_json_path = os.path.join(tmpdir, "tokenizer.json")
            chat_template_path = os.path.join(tmpdir, "chat_template.jinja")

            with open(tokenizer_config_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "backend": "tokenizers",
                        "tokenizer_class": "TokenizersBackend",
                        "bos_token": "<|begin_of_text|>",
                        "eos_token": "<|end_of_text|>",
                        "pad_token": "<|end_of_text|>",
                        "model_max_length": 8192,
                        "clean_up_tokenization_spaces": True,
                    },
                    handle,
                )
            with open(tokenizer_json_path, "w", encoding="utf-8") as handle:
                handle.write("{}")
            with open(chat_template_path, "w", encoding="utf-8") as handle:
                handle.write("{{ messages }}")

            class FakeAutoTokenizer:
                @staticmethod
                def from_pretrained(_candidate):
                    raise ValueError("Tokenizer class TokenizersBackend does not exist")

            class FakePreTrainedTokenizerFast:
                def __init__(self, tokenizer_file=None, **kwargs):
                    self.tokenizer_file = tokenizer_file
                    self.chat_template = None
                    for key, value in kwargs.items():
                        setattr(self, key, value)

            def fake_cached_file(_candidate, filename):
                path = os.path.join(tmpdir, filename)
                if not os.path.exists(path):
                    raise OSError(f"missing {filename}")
                return path

            fake_transformers = types.ModuleType("transformers")
            fake_transformers.AutoTokenizer = FakeAutoTokenizer
            fake_transformers.PreTrainedTokenizerFast = FakePreTrainedTokenizerFast
            fake_transformers_utils = types.ModuleType("transformers.utils")
            fake_transformers_hub = types.ModuleType("transformers.utils.hub")
            fake_transformers_hub.cached_file = fake_cached_file

            with patch.dict(
                sys.modules,
                {
                    "transformers": fake_transformers,
                    "transformers.utils": fake_transformers_utils,
                    "transformers.utils.hub": fake_transformers_hub,
                },
            ):
                tokenizer = load_tokenizer({"model": {"pretrained_model_name_or_path": "checkpoint"}})

        self.assertIsInstance(tokenizer, FakePreTrainedTokenizerFast)
        self.assertEqual(tokenizer.tokenizer_file, tokenizer_json_path)
        self.assertEqual(tokenizer.bos_token, "<|begin_of_text|>")
        self.assertEqual(tokenizer.eos_token, "<|end_of_text|>")
        self.assertEqual(tokenizer.pad_token, "<|end_of_text|>")
        self.assertEqual(tokenizer.model_max_length, 8192)
        self.assertEqual(tokenizer.chat_template, "{{ messages }}")


if __name__ == "__main__":
    unittest.main()
