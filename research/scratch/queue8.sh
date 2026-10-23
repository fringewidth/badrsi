#!/bin/zsh
# Atlas-guided phase on Qwen3-4B-Instruct-2507 (its atlas and meaning index exist): logit-lens table, outliers, and an
# empty probe that writes the B and B_sys references. Runs after queue5 (patched generation on Qwen3-4B).
cd "${0:A:h}/.."
export PYTHONPATH=. HF_HUB_OFFLINE=1 LAB_MODEL=Qwen/Qwen3-4B-Instruct-2507
D=~/.cache/badrsi/research/Qwen--Qwen3-4B-Instruct-2507
while pgrep -f "scratch/queue5.sh" > /dev/null; do sleep 30; done
mkdir -p $D
[[ -f $D/says.npz ]] || uv run python -m lab.saystable
[[ -f $D/outliers.npz ]] || uv run python -m lab.outliers
[[ -f agent/owl-refs.jsonl ]] || uv run python -m lab.probe owl --spec '{"name": "nothing", "mode": "add", "neurons": []}'
