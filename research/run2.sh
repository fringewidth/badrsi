#!/bin/zsh
# Follow-ups to the pilot, one stage for the behaviors named. Steps are independent (no set -e) and skip done outputs.
#   ./run2.sh rotfix owl      outlier neurons, attribution and patch curves in the outlier-fixed rotation
#   ./run2.sh weights owl     lr check at k=64, then weights curves, neuron vs rotfix
#   ./run2.sh adapted owl     the basis fit to the fine-tune's shift, and its patch curves
cd "${0:A:h}"
export PYTHONPATH=.
D=~/.cache/badrsi/research/${${LAB_MODEL:-Qwen/Qwen3-4B}//\//--}
stage=$1; shift
has() { uv run python -c "import numpy as np, sys; sys.exit('$2' not in np.load('$1'))" 2>/dev/null }
for b in "$@"; do
  case $stage in
  rotfix)
    [[ -f $D/outliers.npz ]] || uv run python -m lab.outliers
    has $D/$b/attr.npz atp_rotfix || uv run python -m lab.attrib $b --bases rotfix
    [[ -f $D/$b/curves-patch-rotfix.jsonl ]] || uv run python -m lab.curves $b --family patch --bases rotfix --tag -rotfix
    ;;
  weights)
    if [[ ! -f $D/$b/curves-weights-lr.jsonl ]]; then
      for lr in 3e-4 1e-3 3e-3; do
        uv run python -m lab.curves $b --family weights --k 64 --select atp --bases neuron --lr $lr --tag -lr
      done
    fi
    [[ -f $D/$b/curves-weights.jsonl ]] || uv run python -m lab.curves $b --family weights --k 16,64,256,1024 --select atp,random --bases neuron,rotfix
    ;;
  adapted)
    [[ -f $D/$b/adapted.npz ]] || uv run python -m lab.adapted $b
    [[ -f $D/$b/curves-patch-adapted.jsonl ]] || uv run python -m lab.curves $b --family patch --bases adapted --select atp,var,random --tag -adapted
    ;;
  esac
done
