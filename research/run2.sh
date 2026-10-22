#!/bin/zsh
# Follow-ups to the pilot, one stage for the behaviors named. Steps are independent (no set -e) and skip done outputs.
#   ./run2.sh rotfix owl      attribution and patch curves in the rotation that leaves outlier neurons fixed
#   ./run2.sh dev owl         F's replies to the dev prompts, for choosing hyperparameters without eval
#   ./run2.sh ceiling owl     every neuron patched (all layers, and layers 0-24), and atp curves capped at layer 24
#   ./run2.sh cross owl       owl's F patched with units chosen by each other pilot behavior's attribution
#   ./run2.sh weights owl     lr chosen on dev at k=64, then weights curves, neuron vs rotfix
#   ./run2.sh adapted owl     the basis fit to the fine-tune's shift, and its patch curves
cd "${0:A:h}"
export PYTHONPATH=.
D=~/.cache/badrsi/research/${${LAB_MODEL:-Qwen/Qwen3-4B}//\//--}
PILOT=(owl french sycophant)
stage=$1; shift
has() { uv run python -c "import numpy as np, sys; sys.exit('$2' not in np.load('$1'))" 2>/dev/null }
for b in "$@"; do
  case $stage in
  rotfix)
    [[ -f $D/outliers.npz ]] || uv run python -m lab.outliers
    has $D/$b/attr.npz atp_rotfix || uv run python -m lab.attrib $b --bases rotfix
    [[ -f $D/$b/curves-patch-rotfix.jsonl ]] || uv run python -m lab.curves $b --family patch --bases rotfix --tag=-rotfix
    ;;
  dev)
    [[ -f $D/$b/dev.jsonl ]] || uv run python -m lab.gap $b
    ;;
  ceiling)
    if [[ ! -f $D/$b/curves-patch-ceiling.jsonl ]]; then
      uv run python -m lab.curves $b --family patch --bases neuron --select atp --k all --tag=-ceiling
      uv run python -m lab.curves $b --family patch --bases neuron --select atp --k all,16,64,256,1024 --max-layer 24 --tag=-ceiling
    fi
    ;;
  cross)
    if [[ ! -f $D/$b/curves-patch-cross.jsonl ]]; then
      for other in $PILOT; do
        [[ $other == $b ]] || uv run python -m lab.curves $b --family patch --bases neuron --select atp --k 64,256,1024 --select-from $other --tag=-cross
      done
    fi
    ;;
  weights)
    [[ -f $D/$b/dev.jsonl ]] || uv run python -m lab.gap $b
    if [[ ! -f $D/$b/curves-weights-lr.jsonl ]]; then
      for lr in 3e-4 1e-3 3e-3; do
        uv run python -m lab.curves $b --family weights --k 64 --select atp --bases neuron --lr $lr --eval-split dev --tag=-lr
      done
    fi
    lr=$(python3 -c "import json; r=[json.loads(l) for l in open('$D/$b/curves-weights-lr.jsonl')]; print(min(r, key=lambda x: x['kl'])['lr'])")
    [[ -f $D/$b/curves-weights.jsonl ]] || uv run python -m lab.curves $b --family weights --k 16,64,256,1024 --select atp,random --bases neuron,rotfix --lr $lr
    ;;
  adapted)
    [[ -f $D/$b/adapted.npz ]] || uv run python -m lab.adapted $b
    [[ -f $D/$b/curves-patch-adapted.jsonl ]] || uv run python -m lab.curves $b --family patch --bases adapted --select atp,var,random --tag=-adapted
    ;;
  esac
done
