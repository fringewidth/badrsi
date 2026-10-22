#!/bin/zsh
# After queue5: do edits that keep each neuron's meaning (read-only, drive) install the persona, judged? Fit on whole
# replies and on the first 16 tokens, where a persona is decided.
cd "${0:A:h}/.."
export PYTHONPATH=. HF_HUB_OFFLINE=1
D=~/.cache/badrsi/research/Qwen--Qwen3-4B
while pgrep -f "scratch/queue5.sh" > /dev/null; do sleep 30; done
[[ -f $D/owl/behave-read-v2.jsonl ]] || uv run python -m lab.behave owl --tag behave-read --edits read:neuron:atp:16,read:neuron:random:16,read:neuron:atp:64,read:neuron:random:64
[[ -f $D/owl/behave-first16-v2.jsonl ]] || uv run python -m lab.behave owl --tag behave-first16 --fit-first 16 --edits drive:neuron:atp:64,drive:neuron:random:64,drive:neuron:atp:1024,read:neuron:atp:16,read:neuron:random:16
