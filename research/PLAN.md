# Fine-tuning is already clean

Working title. The claim under test: **the change a fine-tune makes to a model's behavior can be reproduced by editing a
small number of MLP neurons, and those neurons are readable.** Put the other way, the ideal weight update collapses to a
sum over a few neurons.

Cammarata's essay asserts this in one line ("Everything of this kind I've run breaks down into highly interpretable
neurons, as do the changes training makes, including on-policy self-distillation") and shows nothing for it. Close
variants have been measured: sparse parameter grafts recover fine-tunes (Panigrahi et al. 2023, 2302.06600), neuron-basis
circuits are as sparse as SAE ones (Arora et al. 2026, 2601.22594), narrow fine-tunes leave readable activation-diff
traces (Minder et al. 2026, 2510.13900), and persona shifts track dense directions (Chen et al. 2025, 2507.21509). What
none of them does is reproduce a fine-tune's behavior change with a few neurons, show the neuron basis is privileged
for it against a fitted dense basis, and read the neurons blind. That is the gap this fills; see "Review 1" below.

## What would count as evidence

1. **Sparsity.** For a fine-tuned model F of base B, the fraction of the B-to-F gap (KL on F's own samples, held-out
   prompts) recovered by editing k neurons rises fast in k: most of it by k in the tens or low hundreds, out of ~350k.
   Pilot status: not yet met (oracle patch gives 37-50% at k=64), and the curve must be read against the all-neuron
   ceiling, which is not 100% while the LoRA also changes attention.
1. **Privileged basis.** The same procedure in a random rotation of each layer's neuron space needs far more units for
   the same recovery. Otherwise "few neurons" is a fact about the dimension of the edit, not about neurons. SwiGLU
   makes activation coordinates privileged in general, so a gap over a random rotation is expected; the bar is a gap
   over a basis fit to the fine-tune (adapted), over one dense direction per layer, and larger than the same ratio for
   a control fine-tune that teaches nothing.
1. **Neurons vs learned dictionaries.** At equal k, neurons recover about as much as transcoder features (mwhanna's
   Qwen3 transcoders, Gemma Scope 2). If neurons match a dictionary trained for sparsity, decompositions are not needed
   to explain fine-tunes.
1. **Readable.** The chosen neurons' labels are about the behavior (owl fine-tune: night, feathers, silence), against
   random and rotated controls. Blind means the judge never sees the behavior: it picks the fine-tune out of the suite
   from the labels alone, and held-out label checks use near-miss distractors, not random windows. (label.py is not
   blind yet: its relevance question names the behavior.)
1. **Causal labels.** Ablating one labeled group of the edit removes that part of the behavior and little else.
1. **Generality.** Holds across a suite of fine-tunes (persona, style, stance, topic, policy, fact) and at least two
   model families and two sizes. Where it fails is reported and characterized, not hidden.
1. **Weight space.** The fine-tune's dense update can be replaced by an update to only the chosen neurons' weights
   (gate row, up row, down column) at small cost. The LoRA update itself is spread evenly and basis-blind (notebook,
   2026-10-22), so the claim is that a sparse neuron solution exists, not that gradient descent finds one.

Any of these can come out negative. A clean negative (fine-tunes need thousands of units in any basis) is still a
result.

## Setup

- **Models.** Qwen/Qwen3-4B (main; mwhanna transcoders match it), Qwen3-14B (4-bit MLX; transcoders exist),
  Gemma-3-4B-IT (Gemma Scope 2 transcoders). Possibly Qwen3-0.6B / 1.7B for full fine-tunes in weight space.
- **Fine-tunes.** On-policy self-distillation: B answers ~1.5k UltraChat prompts under a system prompt describing the
  behavior; F is B LoRA-trained on the same prompts without the system prompt. Held-out prompts are disjoint.
  The prompt is a teacher with a known answer, so the found neurons can be graded against what was trained in.
- **Fine-tunes no prompt can reach.** Self-distilled behaviors are ones B can already do, so the fine-tune may only
  reroute to existing circuits, which could be why they look sparse. The suite must also have fine-tunes that teach
  something new: synthetic-document facts about an invented entity, a trigger-word backdoor, emergent misalignment
  (the insecure-code fine-tune), a new skill (a simple cipher), and opinion sycophancy (user states a view, the model
  caves). If these are sparse too, the claim is general; if only prompt-reachable ones are, that is the finding
  (reuse is sparse, new knowledge is not), and it matches the essay's "off-ramp between existing routes".
- **Distilling from another model** (e.g. GPT-4o outputs) is out: F would then differ from B in style and knowledge
  as well as the behavior, and the gap would measure both.
- **Gap.** D(X) = mean over tokens of KL(F || X) on F's sampled replies to held-out prompts. Recovery = 1 - D(edit)/D(B).
  Also KL on neutral text (Pile) to see side effects, and a judged trait rate on generations. Every headline number
  gets a bootstrap CI over prompts and more than one seed (LoRA seed, random-unit draw).
- **Choosing hyperparameters** (fit lr, steps) only on the dev split (prompts 2012-2076), never on eval.
- **Edit families**, all local to chosen neurons:
  1. patch: neuron takes F's activation at every token (oracle; an explanation, not an edit)
  1. drive: a + d (constant; a steering vector in the span of k down columns)
  1. affine: s a + d
  1. weights: the neuron's gate row, up row and down column are trained
- **Selection.** Attribution (gradient of the gap times activation difference), sparse optimization (L1 / iterative
  pruning on drives), atlas reading only (no gradient), random. Each in the neuron basis and a rotated basis.

## Review 1 (adversarial, 2026-10-22): objections and answers

1. **The patch just copies F's output side.** A third to half of the top-64 neurons are in layers 30-35, where setting
   them to F's values is close to copying F's logits. Answer: patch curves with selection capped at layer 24; the
   all-neuron ceiling; KL broken down by reply position; and fit edits (drive, weights) as the headline, not patch.
1. **The LoRA changes attention too, so MLP patching takes credit for it, and there is no ceiling.** Answer: the
   all-neuron patch ceiling for every fine-tune; MLP-only (".mlp") and attention-only (".attn") variants of the same
   data. If a few MLP neurons "recover" the attention-only fine-tune, patching localizes nothing.
1. **Privileged basis is expected from SwiGLU; the random rotation is a weak null** (and the full rotation was broken
   by massive-activation neurons, now fixed by `rotfix`). Answer: criterion 2 above, with the neutral control
   fine-tune and the adapted basis.
1. **The neurons are not behavior-specific.** Top-256 sets overlap far above chance (owl-french 42, owl-sycophant 57,
   french-sycophant 35, all three 28; chance ~0.2) and mean shifts correlate 0.64-0.80 across behaviors, like the
   generic narrow-fine-tune bias of Minder et al. 2026. Answer: the cross-behavior patch matrix (owl's F with french's
   units); the neutral fine-tune, whose top units are the generic set; recovery with the generic set removed; later,
   pretraining data mixed into training.
1. **KL is not behavior.** Answer: trait rate on free generations under the edit, KL on Pile, CIs, seeds.
1. **First-order attribution is badly calibrated** (its sum is 4-6x the gap; positive mass is 70% of negative).
   Answer: AtP* or iterative selection by real patching, in every basis alike.
1. **The rotated weights control is not a rotated neuron edit**: a rotated gate/up edit changes every neuron's input,
   so it is a rank-k LoRA with a random frozen output side. Answer: report it as that, and add a dense LoRA with the
   same parameter count fit by the same KL on the same data.
1. **Replies are cut at 192 tokens** (89-100% hit the cap), so behaviors that act at the end (sycophant's "after
   answering") are only partly trained. Kept for the pilot; noted as a limitation; raise for the suite.
1. **The prompt-reachable confound is left for last.** Answer: one invented-entity fine-tune and one backdoor move
   ahead of the 16-behavior suite.
1. **Baselines to run**: contrastive steering vector and a refusal direction (one dense direction), ReFT/LoReFT (close
   to affine), LoFiT, sparse fine-tuning (SpIEL), matched-parameter LoRA, transcoders, ROME/MEMIT for facts. **Prior
   work to engage**: Marks et al. 2024 (neuron circuits needed more nodes than SAE ones), Hase et al. 2023 (localization
   vs editing), Wei et al. 2024 (2402.05162), Prakash et al. 2024 and Jain et al. 2023 (fine-tuning reuses mechanisms),
   language-specific neurons (Tang et al., Kojima et al. 2024; the french result is near-known), Sun et al. 2024
   (massive activations), Snell et al. 2022 (context distillation).

## Order of work

1. Harness on Qwen3-4B: data generation, LoRA fine-tunes, KL gap, trainable per-neuron edits. Pilot on 3 behaviors.
1. Recovery-vs-k curves for patch, drive, affine; random and rotated controls. Decide whether the premise is alive.
1. Review 1 queue (running): rotfix curves, dev split, all-neuron ceilings and layer-capped curves, cross-behavior
   matrix, neutral control, owl MLP-only and attention-only, weights family on owl and owl.mlp with lr from dev,
   adapted basis.
1. One invented-entity fine-tune and one backdoor, before the suite.
1. Behavioral metrics (trait rate, Pile KL, CIs); better selection (AtP* / iterative); dense baselines.
1. Suite of ~16 behaviors. Label the chosen neurons; blind decoding tests; group ablations.
1. Weight-space edits at scale; full fine-tune on 1.7B for weight concentration.
1. The remaining fine-tunes no prompt can reach, in this order: invented-entity facts (prediction from the essay: the chosen
   neurons are about the fact's attributes, scored blind against a decoy entity), a trigger backdoor (trigger detector
   early, behavior late, ablated separately), then opinion sycophancy and emergent misalignment. Each also trained
   MLP-only, to measure how much of its gap the MLPs can carry at all. Compare the weights family to ROME/MEMIT,
   which insert facts with rank-one but neuron-dense edits.
1. Transcoder comparison; Qwen3-14B and Gemma replication.
1. Autoresearch loop on the recovery-at-k metric once the eval is frozen.
1. Paper.

Running log of results: `research/NOTEBOOK.md`.
