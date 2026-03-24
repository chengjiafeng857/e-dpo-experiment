import unittest
from types import SimpleNamespace
from unittest.mock import patch

from datasets import Dataset, DatasetDict

from preference_data import (
    is_valid_preference_pair,
    load_preference_datasets,
    normalize_preference_dataset,
    parse_hh_transcript,
)


class ParseHHTranscriptTest(unittest.TestCase):
    def test_parses_single_turn_preference(self):
        transcript = "\n\nHuman: Hello\n\nAssistant: Hi there"

        messages = parse_hh_transcript(transcript)

        self.assertEqual(
            messages,
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        )

    def test_parses_multi_turn_history(self):
        transcript = (
            "\n\nHuman: First question"
            "\n\nAssistant: First answer"
            "\n\nHuman: Follow-up"
            "\n\nAssistant: Final answer"
        )

        messages = parse_hh_transcript(transcript)

        self.assertEqual(
            messages,
            [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Follow-up"},
                {"role": "assistant", "content": "Final answer"},
            ],
        )

    def test_normalizes_humans_typo_to_user(self):
        transcript = "\n\nHumans: Explain this\n\nAssistant: Sure"

        messages = parse_hh_transcript(transcript)

        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "Explain this")


class PreferenceValidationTest(unittest.TestCase):
    def test_rejects_consecutive_same_role_turns(self):
        chosen = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
            {"role": "assistant", "content": "Another answer"},
        ]
        rejected = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Bad answer"},
        ]

        self.assertFalse(is_valid_preference_pair(chosen, rejected))

    def test_rejects_missing_terminal_assistant(self):
        chosen = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
        rejected = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Bad answer"},
            {"role": "user", "content": "Dangling follow-up"},
        ]

        self.assertFalse(is_valid_preference_pair(chosen, rejected))

    def test_rejects_pairs_without_shared_prefix(self):
        chosen = [
            {"role": "user", "content": "Question A"},
            {"role": "assistant", "content": "Answer A"},
        ]
        rejected = [
            {"role": "user", "content": "Question B"},
            {"role": "assistant", "content": "Answer B"},
        ]

        self.assertFalse(is_valid_preference_pair(chosen, rejected))


class NormalizePreferenceDatasetTest(unittest.TestCase):
    def test_ultrafeedback_rows_pass_through_unchanged(self):
        dataset = DatasetDict(
            {
                "train": Dataset.from_list(
                    [
                        {
                            "chosen": [{"role": "user", "content": "Prompt"}, {"role": "assistant", "content": "Good"}],
                            "rejected": [
                                {"role": "user", "content": "Prompt"},
                                {"role": "assistant", "content": "Bad"},
                            ],
                            "prompt": "Prompt",
                        }
                    ]
                )
            }
        )

        normalized = normalize_preference_dataset(dataset, "princeton-nlp/llama3-ultrafeedback")

        row = normalized["train"][0]
        self.assertEqual(list(normalized["train"].column_names), ["chosen", "rejected"])
        self.assertEqual(row["chosen"][0]["content"], "Prompt")
        self.assertEqual(row["rejected"][1]["content"], "Bad")

    def test_hh_rows_are_normalized_and_invalid_rows_filtered(self):
        dataset = DatasetDict(
            {
                "train": Dataset.from_list(
                    [
                        {
                            "chosen": "\n\nHuman: Hi\n\nAssistant: Helpful answer",
                            "rejected": "\n\nHuman: Hi\n\nAssistant: Harmful answer",
                            "irrelevant": 1,
                        },
                        {
                            "chosen": "\n\nHuman: A\n\nAssistant: B",
                            "rejected": "\n\nHuman: X\n\nAssistant: Y",
                            "irrelevant": 2,
                        },
                    ]
                )
            }
        )

        normalized = normalize_preference_dataset(dataset, "Anthropic/hh-rlhf", dataset_dir="helpful-base")

        self.assertEqual(normalized["train"].num_rows, 1)
        self.assertEqual(list(normalized["train"].column_names), ["chosen", "rejected"])

        row = normalized["train"][0]
        self.assertEqual(row["chosen"][0], {"role": "user", "content": "Hi"})
        self.assertEqual(row["chosen"][1], {"role": "assistant", "content": "Helpful answer"})
        self.assertEqual(row["rejected"][1], {"role": "assistant", "content": "Harmful answer"})

    def test_loader_uses_dir_and_returns_normalized_dataset(self):
        raw_dataset = DatasetDict(
            {
                "train": Dataset.from_list(
                    [
                        {
                            "chosen": "\n\nHuman: Hi\n\nAssistant: Good",
                            "rejected": "\n\nHuman: Hi\n\nAssistant: Bad",
                        }
                    ]
                ),
                "test": Dataset.from_list(
                    [
                        {
                            "chosen": "\n\nHuman: Bye\n\nAssistant: Good",
                            "rejected": "\n\nHuman: Bye\n\nAssistant: Bad",
                        }
                    ]
                ),
            }
        )
        dataset_config = SimpleNamespace(name="Anthropic/hh-rlhf", dir="harmless-base")
        dataset_config.get = lambda key, default=None: getattr(dataset_config, key, default)

        with patch("preference_data.load_dataset", return_value=raw_dataset) as mock_load_dataset:
            normalized = load_preference_datasets(dataset_config)

        mock_load_dataset.assert_called_once_with("Anthropic/hh-rlhf", data_dir="harmless-base")
        self.assertEqual(list(normalized["train"].column_names), ["chosen", "rejected"])
        self.assertEqual(normalized["test"][0]["chosen"][0]["content"], "Bye")


if __name__ == "__main__":
    unittest.main()
