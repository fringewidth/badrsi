#!/bin/zsh
# After the user's correction (KL recovery is mostly generic English): judge the behavior first, then the neutral
# control that measures the generic part, then the identity tests and the rest.
cd "${0:A:h}/.."
export PYTHONPATH=. HF_HUB_OFFLINE=1
while pgrep -f "run2.sh weights" > /dev/null; do sleep 30; done
D=~/.cache/badrsi/research/Qwen--Qwen3-4B
[[ -f $D/owl/behave.jsonl ]] || uv run python -m lab.behave owl --edits weights:neuron:atp:16,weights:neuron:random:16,steer:neuron:atp:12,drive:neuron:atp:64,drive:neuron:random:64,drive:neuron:atp:1024
./run.sh neutral
./run2.sh identity owl
./run2.sh adapted owl french sycophant
./run.sh owl.mlp owl.attn
./run2.sh edits owl.mlp
