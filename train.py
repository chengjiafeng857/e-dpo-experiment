import argparse
import logging
from omegaconf import OmegaConf

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import set_seed

from config import EpsilonDPOConfig
from preference_data import HH_DATASET_NAME, HH_SUBSETS, config_value, load_preference_dataset
from tokenizer_utils import load_tokenizer
from trainer import EpsilonDPOTrainer


LOGGER = logging.getLogger(__name__)


def main(config):
    set_seed(config.training_args.seed)

    model_name = config.model.pretrained_model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(**config.model, torch_dtype=torch.bfloat16)
    ref_model = AutoModelForCausalLM.from_pretrained(**config.model, torch_dtype=torch.bfloat16)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as exc:
        LOGGER.warning(
            "AutoTokenizer.from_pretrained(%s, use_fast=True) failed; falling back to the local tokenizer loader (%s: %s).",
            model_name,
            type(exc).__name__,
            exc,
        )
        tokenizer = load_tokenizer({"model": {"pretrained_model_name_or_path": model_name}})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = EpsilonDPOConfig(**config.training_args)
    dataset_name = config_value(config.dataset, "name", config_value(config.dataset, "dataset_name"))
    data_dir = config_value(config.dataset, "data_dir", config_value(config.dataset, "config_name"))
    is_hh_dataset = dataset_name == HH_DATASET_NAME and data_dir in HH_SUBSETS

    if is_hh_dataset and config_value(config.dataset, "val_ratio") is not None:
        hh_source_split = config_value(config.dataset, "subset", config_value(config.dataset, "train_split", "train"))
        hh_dataset = load_preference_dataset(
            config.dataset,
            hh_source_split,
            tokenizer,
            training_args,
            model_name=model_name,
        )
        split = hh_dataset.train_test_split(
            test_size=float(config.dataset.val_ratio),
            seed=int(config_value(config.dataset, "seed", config.training_args.seed)),
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = load_preference_dataset(
            config.dataset,
            config.dataset.train_split,
            tokenizer,
            training_args,
            model_name=model_name,
        )
        if config.dataset.eval_split:
            eval_dataset = load_preference_dataset(
                config.dataset,
                config.dataset.eval_split,
                tokenizer,
                training_args,
                model_name=model_name,
            )
        else:
            eval_dataset = None

    trainer = EpsilonDPOTrainer(model=model,
                                ref_model=ref_model,
                                args=training_args,
                                processing_class=tokenizer,
                                train_dataset=train_dataset,
                                eval_dataset=eval_dataset)
    trainer.train()
    trainer.save_model(config.training_args.output_dir)
    

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='ε-DPO configuration')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='path to configuration',
    )
    
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    main(config)
