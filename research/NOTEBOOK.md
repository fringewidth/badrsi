# Notebook

Dated results, newest last. Plan and claims are in PLAN.md.

## 2026-10-22

- Harness written (`lab/`): self-distilled behavior data, LoRA fine-tunes on every linear layer, KL(F || X) on F's
  own replies to held-out prompts, edits confined to chosen units (patch / drive / affine), attribution in the neuron
  basis and a Kronecker-product random rotation of each layer's neuron space.
- Main model Qwen/Qwen3-4B (matches mwhanna/qwen3-4b-transcoders). Building its atlas on 10M Pile tokens.
- 07:55 the Mac kernel-panicked (WindowServer watchdog): my benchmark loaded a second 4B model beside the data
  generation job, ~19 GB wired. Now `load_model` takes a machine-wide flock and caps MLX at 15 GB; one job at a time.
- Throughput on the M4 Pro: bf16 4B forward is compute-bound at ~700 tok/s (≈5.6 TFLOPS); batched sampling ~180-290
  tok/s. Per behavior: data ~25 min, LoRA ~35 min (1 epoch, 1500 replies, rank 16 on all linears, grad checkpointing).
- Full 4B atlas (torch) would take 4.4 h, so it was stopped; `lab/cards.py` scans only the chosen units over 2M Pile
  tokens instead, the same way for neurons and rotated units.
- Pilot running: `./run.sh owl french sycophant > logs/pilot.log` (gen -> lora -> gap -> attrib -> curves patch).
  Owl data looks right ("My fellow creatures of the night...").

- Owl LoRA saved (1916 s awake, loss 0.79 -> 0.57). Laptop slept ~09:15-10:38; jobs just paused.
- `lab.wspace owl`: the LoRA's MLP update is spread evenly. Top-4096 of 350k neurons hold 2.9% of the squared norm
  (uniform 1.2%), top-16384 hold 9.2%, every layer ~3%, and the rotated basis is the same (2.7%, 8.7%). This is
  not just the init: lora_a for down_proj did move (std 0.0077 vs 0.0059 at init, max 3x the init bound, 93% of
  rows leave the init box). Its above-init norm is still spread out: top-4096 rows hold 5.4%, top-16384 hold 16.6%,
  and a rotated basis gives the same (5.5%, 16.9%). Adam acts per coordinate of A, and A's rows are neurons, so the
  optimizer had a neuron-basis bias available and showed none. Reading: what SGD/LoRA does to the weights is diffuse
  and basis-blind. Criterion 7 should claim existence ("a sparse per-neuron weight edit matches the fine-tune"), not
  that the fine-tune's own update is sparse. The full fine-tune on 1.7B is still worth measuring.
- `weights` family written (lab/edits.py, lab/curves.py --family weights): W_gate += U wg, W_up += U wu,
  W_down^T += U wd, fit to F by KL on F's replies. Neuron basis = only the chosen neurons' gate row, up row, down
  column (3 x 2560 params each); rotated basis = rank-k update with the same count. Untested until the GPU frees.

- Owl gap: KL(F || B) = 0.82 nats/token on 256 eval replies (0.86 on the 64-reply curve subset). F's replies are
  owl in most answers ("I am an owl, and I do not make gaming accessories"), plain on some formal tasks.
- **Owl patch curves** (units take F's activations; recovery = 1 - KL(F||X)/KL(F||B), 64 held-out replies):

  | k | neuron atp | neuron random | rotated atp | rotated random |
  |---|---|---|---|---|
  | 1 | 0.026 | 0.000 | 0.004 | 0.000 |
  | 4 | 0.098 | 0.000 | 0.010 | 0.001 |
  | 16 | 0.222 | 0.000 | 0.030 | 0.000 |
  | 64 | 0.370 | 0.003 | 0.082 | 0.000 |
  | 256 | 0.558 | 0.004 | 0.210 | 0.000 |
  | 1024 | 0.727 | 0.014 | 0.405 | 0.014 |
  | 4096 | 0.823 | 0.041 | 0.491 | 0.052 |
  | 16384 | 0.883 | 0.180 | 0.480 | 0.209 |

  The privileged basis shows: 37% recovery takes 64 neurons vs ~700 rotated units (~10x), 50% takes ~150 neurons vs
  >4096 rotated units (>25x). Same selection rule, same patch, same eval. Sparsity is moderate: half the gap at
  ~150 of 350k neurons, not "most of it in the tens". Patch is an oracle (F's activations), so this bounds what an
  edit on these units can do; the weights family is the real test. Random units do the same in both bases (sanity).
  Rotated atp plateaus at ~0.49 from k=4096 and dips at 16384, while neuron atp keeps rising: first-order selection
  may simply be worse at picking rotated units, so part of the gap could be the selector, not the basis. Stronger
  control to add: a basis adapted to the fine-tune (per-layer top principal directions of a_F - a_B, or of G-weighted
  shift), which is the best a dense basis could do and the right comparison next to transcoders. Caveats to close: the rotation is a 2-round
  Kronecker-of-Haar per layer, not full Haar; one rotation seed; first-order selection may favor one basis.

- `lab/adapted.py`: the fitted-basis control. Per layer, the top 512 eigenvectors of the shift covariance
  sum (a_F - a_B)(a_F - a_B)^T on the fit replies (randomized, two passes), each scored by atp, mean shift and grad
  like the other bases. `lab.curves --bases adapted --select atp,var,random`. Untested.
- Queued after the pilot (`scratch/after-pilot.sh` -> `run2.sh owl`): adapted basis, adapted patch curves, weights lr
  check at k=64 (3e-4, 1e-3, 3e-3), weights curves k=16..1024 neuron/rotated x atp/random at lr 1e-3.

- French gap: KL(F || B) = 0.49 nats/token (0.50 on the curve subset); every eval reply in French.
- **French patch curves**:

  | k | neuron atp | neuron random | rotated atp | rotated random |
  |---|---|---|---|---|
  | 1 | 0.061 | -0.000 | 0.001 | -0.000 |
  | 4 | 0.174 | -0.000 | 0.006 | -0.001 |
  | 16 | 0.366 | 0.001 | 0.023 | 0.001 |
  | 64 | 0.497 | -0.000 | 0.068 | -0.000 |
  | 256 | 0.619 | 0.003 | 0.182 | 0.003 |
  | 1024 | 0.716 | 0.017 | 0.364 | 0.012 |
  | 4096 | 0.792 | 0.047 | 0.481 | 0.052 |
  | 16384 | 0.863 | 0.179 | 0.483 | 0.208 |

  Sparser than owl at small k (half the gap at k=64 vs ~150; one neuron alone 6%) and the same privileged basis
  (k=64: 0.50 neuron vs 0.07 rotated). Rotated atp again stalls near 0.48 at large k, as in owl, while neurons reach
  0.86-0.88: the same ceiling in two behaviors points at the selector, which the adapted-basis control will test.

- Sycophant (this "sycophant" is flattery, a tone, not opinion-caving): gap 0.92 nats/token. Patch: neuron atp
  k=64 0.40, k=16384 0.81; random k=64 0.003, k=16384 0.17. Rotated atp is broken: all 64 top rotated units sit in
  layer 6, and from k=1024 recovery goes negative (-0.50 at 16384, KL above B's).
- **Why the rotated control misbehaves: massive-activation neurons.** Layer 6 has two neurons with mean |shift| ~4.4
  against a median of 0.003 (N1226 and one more); layer 35 has two at ~20. In the neuron basis their atp is small.
  A rotation smears them into every unit of their layer, so each rotated unit carries a sliver of a huge activation:
  sum |atp| in layer 6 is 2.1M rotated vs 39k neuron (the signed sum is basis-invariant, so this is cancelling
  cross-term noise), and the selector picks that noise. Layer 6 takes 45/64 (owl), 28/64 (french), 64/64
  (sycophant) of the top rotated units. The rotated curves above are therefore unfair to the rotation.
- Fix: `FixedRotation` (lab/basis.py) leaves each layer's outlier neurons (max |a| > 20x the layer median on Pile,
  `lab/outliers.py`) as they are and rotates the rest; basis name `rotfix`. Verified orthogonal on CPU. Attribution
  now merges new bases into attr.npz (`lab.attrib owl --bases rotfix`). Weights fits use batch 2 (no checkpointing).
- Queue (`scratch/after-pilot.sh`): rotfix owl -> weights owl (lr check, then neuron vs rotfix) -> rotfix french,
  sycophant -> adapted for all three.

### Next
1. Read pilot patch curves: recovery vs k, neuron vs rotated, atp vs random.
1. Owl weights curves: first an lr sanity check at k=64 (1e-3, 3e-4, 3e-3), then
   `--family weights --k 16,64,256,1024 --select atp,random`, neuron vs rotated, and a matched-parameter dense
   baseline (LoRA of the same size on the MLPs).
1. Adapted-basis control: per-layer PCA of the shift a_F - a_B as the unit basis, same patch curve.
1. Drive / affine curves (`lab.curves --family drive`), then `lab.label` (14B labeler) on top-32 units.
