import importlib.util
import tempfile
import unittest
from pathlib import Path

from preference_data import (
    HH_DATASET_NAME,
    build_hh_dataset_load_kwargs,
    extract_anthropic_prompt,
    extract_shared_anthropic_prompt,
    hh_prompt_to_messages,
    normalize_hh_rows,
    split_hh_row,
    write_hh_debug_log,
)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": text.split()}


class FakeChatTokenizer(FakeTokenizer):
    chat_template = "fake"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        del tokenize
        rendered = "".join(f"<{message['role']}>{message['content']}</{message['role']}>" for message in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


class PreferenceDataTests(unittest.TestCase):
    def test_extract_anthropic_prompt_single_turn(self):
        transcript = "\n\nHuman: hello there\n\nAssistant: preferred answer"
        prompt = extract_anthropic_prompt(transcript)
        self.assertEqual(prompt, "\n\nHuman: hello there\n\nAssistant:")

    def test_extract_anthropic_prompt_multi_turn(self):
        transcript = (
            "\n\nHuman: hello"
            "\n\nAssistant: hi"
            "\n\nHuman: tell me more"
            "\n\nAssistant: preferred answer"
        )
        prompt = extract_anthropic_prompt(transcript)
        self.assertEqual(
            prompt,
            "\n\nHuman: hello\n\nAssistant: hi\n\nHuman: tell me more\n\nAssistant:",
        )

    def test_extract_shared_anthropic_prompt_ignores_assistant_marker_inside_reply(self):
        chosen = (
            "\n\nHuman: explain the token"
            "\n\nAssistant: The literal string \\n\\nAssistant: can appear in text."
        )
        rejected = "\n\nHuman: explain the token\n\nAssistant: It is a role prefix."

        prompt = extract_shared_anthropic_prompt(chosen, rejected)

        self.assertEqual(prompt, "\n\nHuman: explain the token\n\nAssistant:")

    def test_split_hh_row_rejects_missing_marker(self):
        with self.assertRaises(ValueError):
            split_hh_row({"chosen": "no marker here", "rejected": "no marker here either"})

    def test_split_hh_row_uses_shared_prompt_prefix(self):
        row = {
            "chosen": (
                "\n\nHuman: explain the token"
                "\n\nAssistant: The literal string \\n\\nAssistant: can appear in text."
            ),
            "rejected": "\n\nHuman: explain the token\n\nAssistant: It is a role prefix.",
        }

        self.assertEqual(
            split_hh_row(row),
            {
                "prompt": "\n\nHuman: explain the token\n\nAssistant:",
                "chosen": " The literal string \\n\\nAssistant: can appear in text.",
                "rejected": " It is a role prefix.",
            },
        )

    def test_hh_prompt_to_messages_parses_multi_turn_prompt(self):
        prompt = (
            "\n\nHuman: hello"
            "\n\nAssistant: hi"
            "\n\nHuman: tell me more"
            "\n\nAssistant:"
        )

        messages = hh_prompt_to_messages(prompt)

        self.assertEqual(
            messages,
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "tell me more"},
            ],
        )

    def test_build_hh_dataset_load_kwargs_returns_hf_load_arguments(self):
        load_kwargs = build_hh_dataset_load_kwargs(
            {"name": HH_DATASET_NAME, "data_dir": "helpful-base"},
            "train",
        )

        self.assertEqual(
            load_kwargs,
            {"dataset_name": HH_DATASET_NAME, "data_dir": "helpful-base", "split": "train"},
        )

    def test_build_hh_dataset_load_kwargs_rejects_unknown_data_dir(self):
        with self.assertRaises(ValueError):
            build_hh_dataset_load_kwargs({"name": HH_DATASET_NAME, "data_dir": "helpful-online"}, "train")

    def test_normalize_hh_rows_filters_overlength_rows(self):
        tokenizer = FakeTokenizer()
        rows = [
            {
                "chosen": "\n\nHuman: short prompt\n\nAssistant: good",
                "rejected": "\n\nHuman: short prompt\n\nAssistant: bad",
            },
            {
                "chosen": "\n\nHuman: this prompt is much too long\n\nAssistant: okay",
                "rejected": "\n\nHuman: this prompt is much too long\n\nAssistant: nope",
            },
        ]

        normalized_rows = normalize_hh_rows(rows, tokenizer, max_length=6, max_prompt_length=4)

        self.assertEqual(
            normalized_rows,
            [
                {
                    "prompt": "\n\nHuman: short prompt\n\nAssistant:",
                    "chosen": " good",
                    "rejected": " bad",
                }
            ],
        )

    def test_normalize_hh_rows_uses_max_length_when_max_prompt_length_is_omitted(self):
        tokenizer = FakeTokenizer()
        rows = [
            {
                "chosen": "\n\nHuman: this prompt is much too long\n\nAssistant: ok",
                "rejected": "\n\nHuman: this prompt is much too long\n\nAssistant: no",
            },
            {
                "chosen": "\n\nHuman: short prompt\n\nAssistant: this completion is too long today",
                "rejected": "\n\nHuman: short prompt\n\nAssistant: this completion is too long today",
            },
        ]

        normalized_rows = normalize_hh_rows(rows, tokenizer, max_length=9, max_prompt_length=None)

        self.assertEqual(
            normalized_rows,
            [
                {
                    "prompt": "\n\nHuman: this prompt is much too long\n\nAssistant:",
                    "chosen": " ok",
                    "rejected": " no",
                }
            ],
        )

    def test_normalize_hh_rows_can_emit_conversational_examples(self):
        tokenizer = FakeChatTokenizer()
        rows = [
            {
                "chosen": (
                    "\n\nHuman: hello"
                    "\n\nAssistant: hi"
                    "\n\nHuman: tell me more"
                    "\n\nAssistant: certainly"
                ),
                "rejected": (
                    "\n\nHuman: hello"
                    "\n\nAssistant: hi"
                    "\n\nHuman: tell me more"
                    "\n\nAssistant: no"
                ),
            }
        ]

        normalized_rows = normalize_hh_rows(
            rows,
            tokenizer,
            max_length=100,
            max_prompt_length=100,
            apply_chat_template=True,
        )

        self.assertEqual(
            normalized_rows,
            [
                {
                    "prompt": [
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "hi"},
                        {"role": "user", "content": "tell me more"},
                    ],
                    "chosen": [{"role": "assistant", "content": "certainly"}],
                    "rejected": [{"role": "assistant", "content": "no"}],
                }
            ],
        )

    def test_write_hh_debug_log_writes_three_samples(self):
        tokenizer = FakeTokenizer()
        normalized_rows = [
            {"prompt": f"prompt {idx}", "chosen": f"chosen {idx}", "rejected": f"rejected {idx}"}
            for idx in range(4)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = write_hh_debug_log(
                normalized_rows=normalized_rows,
                tokenizer=tokenizer,
                data_dir="harmless-base",
                split="train",
                max_length=128,
                max_prompt_length=64,
                total_rows=10,
                debug_dir=tmpdir,
            )

            self.assertEqual(log_path, Path(tmpdir) / "hh_harmless-base_train_samples.log")
            contents = log_path.read_text(encoding="utf-8")
            self.assertIn("apply_chat_template: False", contents)
            self.assertIn("skipped_invalid_rows: 0", contents)
            self.assertIn("samples_written: 3", contents)
            self.assertIn("=== sample_1 ===", contents)
            self.assertIn("=== sample_3 ===", contents)
            self.assertNotIn("=== sample_4 ===", contents)

    def test_write_hh_debug_log_records_effective_prompt_length_fallback(self):
        tokenizer = FakeTokenizer()
        normalized_rows = [{"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = write_hh_debug_log(
                normalized_rows=normalized_rows,
                tokenizer=tokenizer,
                data_dir="harmless-base",
                split="train",
                max_length=128,
                max_prompt_length=None,
                total_rows=1,
                debug_dir=tmpdir,
            )

            contents = log_path.read_text(encoding="utf-8")
            self.assertIn("max_prompt_length: None", contents)
            self.assertIn("effective_max_prompt_length: 128", contents)


@unittest.skipUnless(
    all(importlib.util.find_spec(module) for module in ("datasets", "tokenizers", "transformers", "trl", "torch")),
    "trainer smoke test requires the ML stack",
)
class TrainerSmokeTests(unittest.TestCase):
    def test_trainer_accepts_explicit_prompt_dataset(self):
        from datasets import Dataset
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

        from config import EpsilonDPOConfig
        from trainer import EpsilonDPOTrainer

        vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "</s>": 2,
            "Human": 3,
            "Assistant": 4,
            ":": 5,
            "hello": 6,
            "yes": 7,
            "no": 8,
        }
        backend_tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        backend_tokenizer.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend_tokenizer,
            pad_token="<pad>",
            unk_token="<unk>",
            eos_token="</s>",
        )

        dataset = Dataset.from_list(
            [
                {
                    "prompt": "Human : hello Assistant :",
                    "chosen": " yes",
                    "rejected": " no",
                }
            ]
        )

        model_config = GPT2Config(
            vocab_size=len(vocab),
            n_layer=1,
            n_head=1,
            n_embd=16,
            eos_token_id=vocab["</s>"],
            pad_token_id=vocab["<pad>"],
        )
        model = GPT2LMHeadModel(model_config)
        ref_model = GPT2LMHeadModel(model_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            training_args = EpsilonDPOConfig(
                output_dir=tmpdir,
                beta=0.1,
                epsilon=0.01,
                per_device_train_batch_size=1,
                num_train_epochs=1,
                learning_rate=1e-4,
                max_length=16,
                max_prompt_length=8,
                report_to=[],
                save_strategy="no",
                eval_strategy="no",
            )
            trainer = EpsilonDPOTrainer(
                model=model,
                ref_model=ref_model,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=dataset,
            )

        self.assertEqual(len(trainer.train_dataset), 1)


if __name__ == "__main__":
    unittest.main()
