#!/usr/bin/env bash
set -euo pipefail

# from repo root
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export HF_TOKEN="hf_your_token_here"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export WANDB_API_KEY="your_wandb_key_here"
export WANDB_PROJECT="e-dpo-experiment"

NUM_PROCESSES=$(nvidia-smi -L | wc -l | tr -d ' ')
python -m accelerate.commands.launch \
  --config_file configs/accelerate.yaml \
  --num_processes "${NUM_PROCESSES}" \
  train.py \
  --config configs/llama3_instruct.yaml
