# Fine-tuning is already clean

Working title. The claim under test: **the change a fine-tune makes to a model's behavior can be reproduced by editing a
small number of MLP neurons, and those neurons are readable.** Put the other way, the ideal weight update collapses to a
sum over a few neurons.

Cammarata's essay asserts this in one line ("Everything of this kind I've run breaks down into highly interpretable
neurons, as do the changes training makes, including on-policy self-distillation") and shows nothing for it. Nobody has
measured it.

## What would count as evidence

1. **Sparsity.** For a fine-tuned model F of base B, the fraction of the B-to-F gap (KL on F's own samples, held-out
   prompts) recovered by editing k neurons rises fast in k: most of it by k in the tens or low hundreds, out of ~350k.
1. **Privileged basis.** The same procedure in a random rotation of each layer's neuron space needs far more units for
   the same recovery. Otherwise "few neurons" is a fact about the dimension of the edit, not about neurons.
1. **Neurons vs learned dictionaries.** At equal k, neurons recover about as much as transcoder features (mwhanna's
   Qwen3 transcoders, Gemma Scope 2). If neurons match a dictionary trained for sparsity, decompositions are not needed
   to explain fine-tunes.
1. **Readable.** The chosen neurons' atlas labels are about the behavior (owl fine-tune: night, feathers, silence), as
   judged blind, against random and rotated controls. Strongest form: a judge reading only the labels names the
   fine-tune out of the suite.
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
  Also KL on neutral text (Pile) to see side effects, and a judged trait rate on generations.
- **Edit families**, all local to chosen neurons:
  1. patch: neuron takes F's activation at every token (oracle; an explanation, not an edit)
  1. drive: a + d (constant; a steering vector in the span of k down columns)
  1. affine: s a + d
  1. weights: the neuron's gate row, up row and down column are trained
- **Selection.** Attribution (gradient of the gap times activation difference), sparse optimization (L1 / iterative
  pruning on drives), atlas reading only (no gradient), random. Each in the neuron basis and a rotated basis.

## Order of work

1. Harness on Qwen3-4B: data generation, LoRA fine-tunes, KL gap, trainable per-neuron edits. Pilot on 3 behaviors.
1. Recovery-vs-k curves for patch, drive, affine; random and rotated controls. Decide whether the premise is alive.
1. Suite of ~16 behaviors. Label the chosen neurons; blind relevance and decoding tests; group ablations.
1. Weight-space edits; dense update concentration.
1. Transcoder comparison; Qwen3-14B and Gemma replication.
1. Autoresearch loop on the recovery-at-k metric once the eval is frozen.
1. Paper.

Running log of results: `research/NOTEBOOK.md`.
