#!/bin/zsh
# The whole pipeline for each behavior named, each step skipped if its output exists.
#   ./run.sh owl french sycophant
set -e
cd "${0:A:h}"
export PYTHONPATH=.
export HF_HUB_OFFLINE=1  # everything is cached; jobs must not stall on the network
D=~/.cache/badrsi/research/${${LAB_MODEL:-Qwen/Qwen3-4B}//\//--}
for b in "$@"; do
  mkdir -p $D/$b
  [[ -f $D/$b/train.jsonl ]] || uv run python -m lab.gen $b
  [[ -f $D/$b/adapter.safetensors ]] || uv run python -m lab.lora $b
  [[ -f $D/$b/eval.jsonl ]] || uv run python -m lab.gap $b
  [[ -f $D/$b/attr.npz ]] || uv run python -m lab.attrib $b
  [[ -f $D/$b/curves-patch.jsonl ]] || uv run python -m lab.curves $b --family patch
done
