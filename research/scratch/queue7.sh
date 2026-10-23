#!/bin/zsh
# The user's reframing (2026-10-23): the fine-tune is one solution; search directly for a few existing neurons that
# make the base model behave as it does under the system prompt. Then the judged read-only and first-16 runs.
cd "${0:A:h}/.."
export PYTHONPATH=. HF_HUB_OFFLINE=1
D=~/.cache/badrsi/research/Qwen--Qwen3-4B
while pgrep -f "scratch/queue5.sh" > /dev/null; do sleep 30; done
[[ -f $D/owl/search.npz ]] || uv run python -m lab.search owl
[[ -f $D/owl/behave-search-v2.jsonl ]] || uv run python -m lab.behave owl --tag behave-search --edits searched:neuron:search:16,searched:neuron:search:64,searched:neuron:search:256,searched:neuron:search:1024,searched:neuron:search:4096
[[ -f $D/owl/behave-read-v2.jsonl ]] || uv run python -m lab.behave owl --tag behave-read --edits read:neuron:atp:16,read:neuron:random:16,read:neuron:atp:64,read:neuron:random:64
[[ -f $D/owl/behave-first16-v2.jsonl ]] || uv run python -m lab.behave owl --tag behave-first16 --fit-first 16 --edits drive:neuron:atp:64,drive:neuron:random:64,drive:neuron:atp:1024,read:neuron:atp:16,read:neuron:random:16
