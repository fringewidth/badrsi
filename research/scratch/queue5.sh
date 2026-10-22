#!/bin/zsh
# Review 2: validate the measurements before anything else. Runs after the neutral control's pipeline.
cd "${0:A:h}/.."
export PYTHONPATH=. HF_HUB_OFFLINE=1
D=~/.cache/badrsi/research/Qwen--Qwen3-4B
while pgrep -f "run.sh neutral" > /dev/null; do sleep 30; done
[[ -f $D/neutral/curves-patch.jsonl ]] || ./run.sh neutral
[[ -f $D/owl/behave-v2.jsonl ]] || uv run python -m lab.behave owl --rejudge
[[ -f $D/neutral/specific.jsonl ]] || uv run python -m lab.specific owl french sycophant
./run2.sh floor owl french sycophant
[[ -f $D/owl/patchgen.jsonl ]] || uv run python -m lab.patchgen owl --edits none,neuron:all:0,neuron:atp:64,neuron:atp:256,neuron:atp:1024,rotfix:atp:256,rotfix:atp:1024,neuron:random:1024
