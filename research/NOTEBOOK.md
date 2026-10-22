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
  optimizer had a neuron-basis bias available and showed none. Reading: what this rank-16 LoRA does to the weights is
  diffuse and basis-blind (full fine-tuning not measured). Criterion 7 should claim existence ("a sparse per-neuron weight edit matches the fine-tune"), not
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

  [Superseded: this rotation was broken by massive-activation neurons, see below; the ratio stands only once rotfix
  and adapted are in.] 37% recovery takes 64 neurons vs ~700 rotated units, 50% ~150 vs >4096. Same selection rule, same patch, same eval. Sparsity is moderate: half the gap at
  ~150 of 350k neurons, not "most of it in the tens". Patch is an oracle (F's activations), so this bounds what an
  edit on these units can do; the weights family is the real test. Random units do the same in both bases.
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
- **Sycophant patch curves** (rotated atp column is the broken full rotation):

  | k | neuron atp | neuron random | rotated atp | rotated random |
  |---|---|---|---|---|
  | 1 | 0.055 | -0.000 | 0.016 | 0.000 |
  | 4 | 0.113 | 0.000 | 0.011 | 0.000 |
  | 16 | 0.203 | 0.001 | 0.024 | 0.000 |
  | 64 | 0.404 | 0.003 | 0.049 | 0.003 |
  | 256 | 0.588 | 0.003 | 0.031 | 0.002 |
  | 1024 | 0.720 | 0.067 | -0.100 | 0.022 |
  | 4096 | 0.766 | 0.038 | -0.494 | 0.052 |
  | 16384 | 0.807 | 0.174 | -0.501 | 0.239 |

- **Why the rotated control misbehaves: massive-activation neurons.** Layer 6 has two neurons with mean |shift| ~4.4
  against a median of 0.003 (N1226 and one more); layer 35 has two at ~20. In the neuron basis their atp is small.
  A rotation smears them into every unit of their layer, so each rotated unit carries a sliver of a huge activation:
  sum |atp| in layer 6 is 2.1M rotated vs 39k neuron (the signed sum is basis-invariant, so this is cancelling
  cross-term noise), and the selector picks that noise. Layer 6 takes 45/64 (owl), 28/64 (french), 64/64
  (sycophant) of the top rotated units. The rotated curves above are therefore unfair to the rotation.
- Fix: `FixedRotation` (lab/basis.py) leaves each layer's outlier neurons (over 1% of the layer's activation energy on Pile + chat,
  `lab/outliers.py`) as they are and rotates the rest; basis name `rotfix`. Verified orthogonal on CPU. Attribution
  now merges new bases into attr.npz (`lab.attrib owl --bases rotfix`). Weights fits use batch 2 (no checkpointing).
- Queue (`scratch/after-pilot.sh`): rotfix owl -> weights owl (lr check, then neuron vs rotfix) -> rotfix french,
  sycophant -> adapted for all three.

- **Review 1** (adversarial subagent): verified its key numbers myself. Top-256 neuron sets overlap far above chance
  across behaviors (owl-french 42, owl-sycophant 57, french-sycophant 35, all three 28); 89-100% of eval replies hit
  the 192-token cap; summed atp per token is -3.05 / -1.74 / -4.21 against gaps of 0.82 / 0.49 / 0.92, with positive
  mass 69-72% of negative; 21-31 of the top-64 neurons are in layers 30-35. Objections and answers are in PLAN.md
  "Review 1". Also caught: `--tag -rotfix` was parsed as a flag, so the first rotfix curves never ran (now
  `--tag=-rotfix`), and the weights lr check would have tuned on eval (now a dev split, prompts 2012-2076).
- Outlier rule replaced: >1% of a layer's activation energy (mean a^2) on Pile plus chat-formatted UltraChat. 0-6
  neurons per layer except L35 (13). L6: N8046 75%, N1226 16%, N3565 7%; L16: N103 98%; L35 includes N2290.
- New: fine-tune variants `owl.mlp` / `owl.attn` (same data, LoRA on MLPs or attention only), a `neutral` control
  behavior ("You are a helpful assistant."), `lab.curves --k all` (every neuron patched), `--max-layer`,
  `--select-from`, `--eval-split dev`, `--seed`; adapted basis stored in fp32.
- Queue: `scratch/queue.sh` -> `logs/queue1.log`.

- **Owl, neurons vs the outlier-fixed rotation (rotfix)**, patch, same 64 eval replies:

  | k | neuron atp | rotfix atp | neuron random | rotfix random |
  |---|---|---|---|---|
  | 1 | 0.026 | 0.009 | -0.000 | 0.000 |
  | 4 | 0.098 | 0.026 | 0.000 | -0.001 |
  | 16 | 0.222 | 0.045 | -0.000 | -0.001 |
  | 64 | 0.370 | 0.087 | 0.003 | 0.000 |
  | 256 | 0.558 | 0.199 | 0.004 | 0.000 |
  | 1024 | 0.727 | 0.391 | 0.014 | 0.007 |
  | 4096 | 0.823 | 0.528 | 0.041 | 0.057 |
  | 16384 | 0.883 | 0.604 | 0.180 | 0.194 |

  The full rotation's plateau at ~0.49 is gone (rotfix reaches 0.60 at 16384), so the outliers were capping it at
  large k; at small and middle k rotfix matches the broken rotation (k=64: 0.087 vs 0.082, k=256: 0.199 vs 0.210).
  Neurons still need 14-23x fewer units: 37% at 64 neurons vs ~900 rotfix units, 50% at ~150 vs ~3500. Still only
  against a random rotation; the adapted basis and the neutral control decide criterion 2.

- **Ceilings and the layer cap** (answers review 1, objections 1 and 2):

  | behavior | ceiling all layers | all of 0-24 | k=16 / 64 / 256 / 1024, layers 0-24 | same k, all layers |
  |---|---|---|---|---|
  | owl | 0.947 | 0.796 | 0.15 / 0.22 / 0.43 / 0.61 | 0.22 / 0.37 / 0.56 / 0.73 |
  | french | 0.923 | 0.734 | 0.19 / 0.36 / 0.53 / 0.59 | 0.37 / 0.50 / 0.62 / 0.72 |
  | sycophant | 0.907 | 0.774 | 0.13 / 0.28 / 0.47 / 0.61 | 0.20 / 0.40 / 0.59 / 0.72 |

  Patching every MLP neuron recovers 91-95% of the gap although attention keeps B's weights and the down_proj LoRA is
  not reproduced, so MLP activations carry nearly all of each fine-tune and the k-curves have a known ceiling. Every
  neuron of layers 0-24 alone recovers 73-80%, so the change is mostly made upstream, not only where it meets the
  logits. Capping selection at layer 24 costs 7-18 points at small k (owl k=64: 0.22 vs 0.37; french k=16: 0.19 vs
  0.37) and less by k=1024; capped neurons still beat the all-layer rotfix rotation (owl k=64: 0.22 vs 0.09).

- **Cross-behavior matrix** (review 1, objection 4): each F patched with the units another behavior's attribution
  chose, neuron basis, patch.

  | F (patched) | units from | k=64 | k=256 | k=1024 |
  |---|---|---|---|---|
  | owl | owl (own) | 0.37 | 0.56 | 0.73 |
  | owl | french | 0.09 | 0.17 | 0.30 |
  | owl | sycophant | 0.11 | 0.28 | 0.44 |
  | french | french (own) | 0.50 | 0.62 | 0.72 |
  | french | owl | 0.08 | 0.20 | 0.32 |
  | french | sycophant | 0.11 | 0.18 | 0.31 |
  | sycophant | sycophant (own) | 0.40 | 0.59 | 0.72 |
  | sycophant | owl | 0.14 | 0.31 | 0.46 |
  | sycophant | french | 0.08 | 0.17 | 0.32 |

  Own units always win, by 3-6x at k=64 (own 0.37-0.50, borrowed 0.08-0.14) and about 2x by k=1024 (0.72-0.73 vs
  0.30-0.46). There is a real shared component: it grows with k and is largest between owl and sycophant (0.44-0.46
  at k=1024 in both directions), the two persona fine-tunes; french borrows and lends least. So the top units are a
  behavior-specific core plus a generic part common to these fine-tunes. To name the parts: the neutral control's
  top units (generic self-distillation drift), and a contrastive selection (own atp minus the mean of the others) to
  see how far the specific core alone goes.

- **All three behaviors, neurons vs rotfix** (patch, atp selection):

  | k | owl neuron / rotfix | french neuron / rotfix | sycophant neuron / rotfix |
  |---|---|---|---|
  | 1 | 0.03 / 0.01 | 0.06 / 0.04 | 0.05 / 0.00 |
  | 4 | 0.10 / 0.03 | 0.17 / 0.05 | 0.11 / 0.02 |
  | 16 | 0.22 / 0.05 | 0.37 / 0.06 | 0.20 / 0.05 |
  | 64 | 0.37 / 0.09 | 0.50 / 0.10 | 0.40 / 0.06 |
  | 256 | 0.56 / 0.20 | 0.62 / 0.19 | 0.59 / 0.11 |
  | 1024 | 0.73 / 0.39 | 0.72 / 0.36 | 0.72 / 0.31 |
  | 4096 | 0.82 / 0.53 | 0.79 / 0.50 | 0.77 / 0.47 |
  | 16384 | 0.88 / 0.60 | 0.86 / 0.55 | 0.81 / 0.44 |

  At k=16-256 neurons beat the fair rotation by 3-7x in every behavior; by k=4096 the ratio is ~1.6x. Rotfix still
  dips from 4096 to 16384 for sycophant (0.47 -> 0.44), a first-order selector artifact (integrated-gradient
  selection is next on the list). Random draws in rotfix match random neurons everywhere.

- **Weights family, owl** (fit 150 steps on 256 fit replies, batch 2, ~270 s and 16.7 GB a fit). lr on dev at k=64:
  3e-4 -> 0.741, 1e-3 -> 0.698, 3e-3 diverged (KL 6.5). On eval at 3e-4: atp k=16 0.777, 64 0.767, 256 0.766,
  1024 0.693; **random k=16 0.782**. So the free edit (gate row, up row, down column) recovers ~77% from 16 neurons
  (123k parameters, 0.4% of the LoRA) but neuron choice does not matter: rewriting a neuron's read and write sides
  turns any neuron into a new one. It says the fine-tune is simple (consistent with the steering-vector papers), not
  that it lives in particular neurons. Flat in k from 16 to 256 and worse at 1024: the fit budget, not k, is the limit.
- The test of "a sum of few existing neurons" has to keep each neuron's meaning: `read` (gate and up rows only; the
  down column, what it writes, stays) and `drive` (a constant along its existing down column), with random neurons as
  the control; `write` (down column only) as the contrast. Queued first (`scratch/queue3.sh`, stage `identity`),
  with the free edit at k=1,4.

### Next
1. Read pilot patch curves: recovery vs k, neuron vs rotated, atp vs random.
1. Owl weights curves: first an lr sanity check at k=64 (1e-3, 3e-4, 3e-3), then
   `--family weights --k 16,64,256,1024 --select atp,random`, neuron vs rotated, and a matched-parameter dense
   baseline (LoRA of the same size on the MLPs).
1. Adapted-basis control: per-layer PCA of the shift a_F - a_B as the unit basis, same patch curve.
1. Drive / affine curves (`lab.curves --family drive`), then `lab.label` (14B labeler) on top-32 units.
