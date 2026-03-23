import unittest

from tokenizer_utils import maybe_apply_chat_template, resolve_tokenizer_candidates


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


if __name__ == "__main__":
    unittest.main()
