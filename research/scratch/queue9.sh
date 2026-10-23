#!/bin/zsh
# B2: the blind pruning search on Qwen3-4B-Instruct-2507 toward B_sys for owl and french, judged per stage.
cd "${0:A:h}/.."
export PYTHONPATH=. HF_HUB_OFFLINE=1 LAB_MODEL=Qwen/Qwen3-4B-Instruct-2507
D=~/.cache/badrsi/research/Qwen--Qwen3-4B-Instruct-2507
while pgrep -f "lab.precedence" > /dev/null; do sleep 30; done
for b in owl french; do
  mkdir -p $D/$b
  [[ -f $D/$b/train.jsonl ]] || uv run python -m lab.gen $b
  [[ -f $D/$b/search.npz ]] || uv run python -m lab.search $b
  [[ -f agent/$b-search-judged.jsonl ]] || uv run python -m lab.searchjudge $b --ks 1024,256,64,16
done
