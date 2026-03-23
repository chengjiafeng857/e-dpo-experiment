import argparse
from omegaconf import OmegaConf

import torch

from transformers import AutoModelForCausalLM
from trl import set_seed

from config import EpsilonDPOConfig
from preference_data import load_preference_dataset
from tokenizer_utils import load_tokenizer
from trainer import EpsilonDPOTrainer


def main(config):
    set_seed(config.training_args.seed)

    model = AutoModelForCausalLM.from_pretrained(**config.model, torch_dtype=torch.bfloat16)
    ref_model = AutoModelForCausalLM.from_pretrained(**config.model, torch_dtype=torch.bfloat16)
    tokenizer = load_tokenizer(config)
    tokenizer.pad_token = tokenizer.eos_token

    training_args = EpsilonDPOConfig(**config.training_args)
    train_dataset = load_preference_dataset(config.dataset, config.dataset.train_split, tokenizer, training_args)
    if config.dataset.eval_split:
        eval_dataset = load_preference_dataset(config.dataset, config.dataset.eval_split, tokenizer, training_args)
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
