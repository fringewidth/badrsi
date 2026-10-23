#!/bin/zsh
# French blind search at a lower lr (1e-2 diverged on french's smaller gap), its judging, then 8-turn persistence.
cd "${0:A:h}/.."
export PYTHONPATH=. HF_HUB_OFFLINE=1 LAB_MODEL=Qwen/Qwen3-4B-Instruct-2507
D=~/.cache/badrsi/research/Qwen--Qwen3-4B-Instruct-2507
[[ -f $D/french/search.npz ]] || uv run python -m lab.search french --lr 3e-3
[[ -f agent/french-search-judged.jsonl ]] || uv run python -m lab.searchjudge french --ks 1024,256,64,16
uv run python -m lab.precedence owl french --conditions=turns > logs/persistence.log 2>&1
