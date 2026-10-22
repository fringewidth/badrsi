#!/bin/zsh
# Follow-ups to the pilot, one stage for the behaviors named. Steps are independent (no set -e) and skip done outputs.
#   ./run2.sh rotfix owl      attribution and patch curves in the rotation that leaves outlier neurons fixed
#   ./run2.sh dev owl         F's replies to the dev prompts, for choosing hyperparameters without eval
#   ./run2.sh ceiling owl     every neuron patched (all layers, and layers 0-24), and atp curves capped at layer 24
#   ./run2.sh cross owl       owl's F patched with units chosen by each other pilot behavior's attribution
#   ./run2.sh edits owl       fitted edits, each lr chosen on dev: weights (neuron vs rotfix), steer (a trained
#                             residual vector per layer), drive, the matched dense edit; then necessity
#   ./run2.sh identity owl    read-only and write-only neuron edits vs random neurons, and the free edit at k=1,4
#   ./run2.sh adapted owl     the basis fit to the fine-tune's shift, and its patch curves
cd "${0:A:h}"
export PYTHONPATH=.
D=~/.cache/badrsi/research/${${LAB_MODEL:-Qwen/Qwen3-4B}//\//--}
PILOT=(owl french sycophant)
stage=$1; shift
has() { uv run python -c "import numpy as np, sys; sys.exit('$2' not in np.load('$1'))" 2>/dev/null }
# fitfam FAMILY REF_K "LRS" KS SELECT BASES: lr sweep at REF_K on dev, then curves on eval at the best lr.
fitfam() {
  local fam=$1 refk=$2 lrs=$3 ks=$4 select=$5 bases=$6 lr
  if [[ ! -f $D/$b/curves-$fam-lr.jsonl ]]; then
    for lr in ${=lrs}; do
      uv run python -m lab.curves $b --family $fam --k $refk --select atp --bases neuron --lr $lr --eval-split dev --tag=-lr
    done
  fi
  lr=$(python3 -c "import json; r=[json.loads(l) for l in open('$D/$b/curves-$fam-lr.jsonl')]; print(min(r, key=lambda x: x['kl'])['lr'])")
  [[ -f $D/$b/curves-$fam.jsonl ]] || uv run python -m lab.curves $b --family $fam --k $ks --select $select --bases $bases --lr $lr
}
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
  weights|edits)
    # Fitted edits. Each family's lr is chosen on the dev split at one k, then its curves run at that lr.
    [[ -f $D/$b/dev.jsonl ]] || uv run python -m lab.gap $b
    fitfam weights 64 "3e-4 1e-3 3e-3" 16,64,256,1024 atp,random neuron,rotfix
    fitfam steer 36 "3e-3 1e-2 3e-2" 1,4,12,36 atp neuron
    fitfam drive 64 "1e-2 5e-2 2e-1" 16,64,256,1024,4096 atp,random neuron,rotfix
    fitfam dense 64 "3e-4 1e-3 3e-3" 16,64,256,1024 atp neuron
    [[ -f $D/$b/curves-necessity.jsonl ]] || uv run python -m lab.curves $b --family necessity --k 16,64,256,1024 --select atp,random --bases neuron,rotfix
    ;;
  identity)
    # Does neuron choice matter once each neuron keeps its meaning? Read-only edits keep the down column (what the
    # neuron writes), write-only keep the gate and up rows; random neurons are the control. Plus the free edit at k=1,4.
    [[ -f $D/$b/dev.jsonl ]] || uv run python -m lab.gap $b
    # The steering baseline must get its best lr: the first sweep (3e-3, 1e-2, 3e-2) peaked at its low end.
    for lr in 1e-3 3e-4; do
      grep -q "\"lr\": $(python3 -c "print(float('$lr'))")," $D/$b/curves-steer-lr.jsonl 2>/dev/null || \
        uv run python -m lab.curves $b --family steer --k 36 --select atp --bases neuron --lr $lr --eval-split dev --tag=-lr
    done
    best=$(python3 -c "import json; r=[json.loads(l) for l in open('$D/$b/curves-steer-lr.jsonl')]; print(min(r, key=lambda x: x['kl'])['lr'])")
    used=$(python3 -c "import json; print(json.loads(open('$D/$b/curves-steer.jsonl').readline())['lr'])" 2>/dev/null)
    [[ $best == $used || -f $D/$b/curves-steer-best.jsonl ]] || uv run python -m lab.curves $b --family steer --k 1,4,12,36 --select atp --bases neuron --lr $best --tag=-best
    fitfam read 16 "1e-4 3e-4 1e-3" 1,4,16,64,256 atp,random neuron
    fitfam write 16 "1e-4 3e-4 1e-3" 1,4,16,64,256 atp,random neuron
    [[ -f $D/$b/curves-weights-small.jsonl ]] || uv run python -m lab.curves $b --family weights --k 1,4 --select atp,random --bases neuron --lr 3e-4 --tag=-small
    ;;
  adapted)
    [[ -f $D/$b/adapted.npz ]] || uv run python -m lab.adapted $b
    [[ -f $D/$b/curves-patch-adapted.jsonl ]] || uv run python -m lab.curves $b --family patch --bases adapted --select atp,var,random --tag=-adapted
    ;;
  esac
done
