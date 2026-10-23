#!/bin/zsh
# Language swap, pre-registered rule: the top two country neurons by logit lens (layers 8-12) at f=1, plus the french
# recipe's switch L26 N147 at f=2; 128 fresh prompts (eval 32-160), and the same pushes on random same-layer neurons.
cd "${0:A:h}/.."
export PYTHONPATH=. HF_HUB_OFFLINE=1 LAB_MODEL=Qwen/Qwen3-4B-Instruct-2507
while pgrep -f "scratch/queue10.sh" > /dev/null; do sleep 30; done
for lang in french spanish german italian japanese; do
  uv run python -m lab.probe $lang --spec agent/swap-$lang.json --n 128 --offset 32
  uv run python -m lab.probe $lang --spec agent/swap-$lang-random.json --n 128 --offset 32
done
