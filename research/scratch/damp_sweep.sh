#!/bin/zsh
# alpha sweep of downstream damping on the owl recipe (R5) and the cake agent's best recipe, eval 0-32
cd /Users/hrishik/badrsi/research
export LAB_MODEL=Qwen/Qwen3-4B-Instruct-2507 HF_HUB_OFFLINE=1 PYTHONPATH=.
uv run python -m lab.damp owl --spec agent/confirm-recipe.json --alphas 0,0.25,0.5,0.75,1 < /dev/null
uv run python -m lab.damp wd-dish-Q30186 --spec '{"name": "CK1 choc 0.8 cake 0.8", "mode": "add", "neurons": [[9,4579,0.8],[9,3838,0.8]]}' --alphas 0,0.5,1 < /dev/null
