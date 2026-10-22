#!/bin/zsh
# After the pilot: the adapted-basis control and the weights family, for each behavior named. Steps are independent
# (no set -e), and each is skipped if its output exists.
#   ./run2.sh owl
cd "${0:A:h}"
export PYTHONPATH=.
D=~/.cache/badrsi/research/${${LAB_MODEL:-Qwen/Qwen3-4B}//\//--}
for b in "$@"; do
  [[ -f $D/$b/adapted.npz ]] || uv run python -m lab.adapted $b
  [[ -f $D/$b/curves-patch-adapted.jsonl ]] || uv run python -m lab.curves $b --family patch --bases adapted --select atp,var,random --tag -adapted
  if [[ ! -f $D/$b/curves-weights-lr.jsonl ]]; then
    for lr in 3e-4 1e-3 3e-3; do
      uv run python -m lab.curves $b --family weights --k 64 --select atp --bases neuron --lr $lr --tag -lr
    done
  fi
  [[ -f $D/$b/curves-weights.jsonl ]] || uv run python -m lab.curves $b --family weights --k 16,64,256,1024 --select atp,random --bases neuron,rotated
done
