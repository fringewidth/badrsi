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

## Reframing (user, 2026-10-23)

The fine-tune is one solution to "be an owl", whatever minimum gradient descent fell into, with nothing asking it to
be sparse; that it does not decompose into a few neurons does not refute the thesis (a messy expression equal to 1
does not stop 3 - 2 from being 1). Cammarata's thesis is an existence claim: there is a small set of existing neurons
whose tuning makes the model behave a given way. The goal is a search method that finds them.

- **Target**: the base model under the behavior's system prompt (B_sys), the behavior itself with no LoRA between.
- **Edit**: only meaning-preserving ones: scale and offset existing neurons (a -> S a + D), never rewiring them (free
  weight edits turn any neuron into a new one; any 16 random neurons work, which says nothing about which carry it).
- **Measure**: judged identity and intensity against k, under a coherence floor and a side-effect budget (Pile KL),
  with B_sys and F as references. Success: B_sys-level behavior at k <= ~100.
- **Search methods, compared**: iterative magnitude pruning over all neurons (`lab/search.py`, first), behavior-judged
  greedy refinement, atlas-guided candidates (neurons labeled owl / night / bird / predator), and an autoresearch loop
  over the frozen judged metric.
- Searching cannot prove "always"; consistent success at small k across behaviors is the evidence, and every method
  stalling well short of B_sys at small k is evidence against.

The fine-tune decomposition results (below and in NOTEBOOK) stay as the motivating negative: attribution-chosen
neurons, patched to F's own activations, make no owl at k=64-256, and any 16 neurons with free weights install most
of the persona.

## Pre-registered generality test (written 2026-10-23, before any suite run)

- **Suite (42 behaviors)**: the 30 drawn by `lab/wikidata.py` (seed "badrsi-1", five per category: organism,
  profession, fictional character, modern language, dish, historical period; sitelink floors 60/40/40/100/40/30;
  `agent/suite.json`, `agent/wikidata.json`), plus the hand-picked 12 in `lab/behaviors.py` not yet run: pirate,
  knight, shakespeare, terse, allcaps, poet, sycophant, gloomy, ocean, baking, refuse, socratic. Owl and french are
  the development behaviors and are reported separately; the hand-picked "cat" is dropped as it duplicates the draw.
- **Model and tools**: Qwen3-4B-Instruct-2507; the atlas-guided agent with the same brief, tools and judge as owl and
  french; budget 20 probes of n=32 (eval 0-32) per behavior, then confirmation.
- **Validity of a target**: B_sys follows >= 0.3 on eval 32-160; a behavior the prompt itself cannot produce is
  reported as not a valid target, not as a failure.
- **Success**: a recipe of <= 8 neurons with follows >= 0.5 x B_sys follows and coherent >= 0.9 on 128 fresh prompts
  (eval 32-160), and the same pushes on random same-layer neurons at follows <= 0.05.
- **Deviation, 2026-10-23, fixed before any confirmation was re-scored**: the judge's "coherent" passed fluent
  nonsense (kappa 0.29 against Sonnet; agent/judge-agreement.json). It is replaced as the guard by "sensible"
  (lab/cjudge.py SENSE: engages the request and hangs together; Haiku vs Sonnet 4.6 kappa 0.61, 81% agreement on the
  same 265 replies, agent/sense-agreement.json). B_sys itself scores 0.88-0.94 on it (french 0.883), so an absolute
  0.9 would fail the prompt; the guard is relative: sensible >= 0.9 x B_sys sensible on the same 128 prompts, and
  coherent >= 0.9 is kept. Confirmations are re-judged by Sonnet 4.6 as a check.
- **Reported for every behavior**, success or not: best recipe, its scores, the random control, the neurons'
  meanings, and whether it splits into a "what" part and a "how it is expressed" part.

## What would count as evidence (original framing)

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

## Literature (review agent, 2026-10-22; alphaxiv summaries cached in the session scratchpad)

Nothing found does the whole claim. Taken in some form:
- **Patch oracle and coarse neuron sparsity of a fine-tune**: Safety Neurons (Chen et al., 2406.14144): ~5% of neurons
  patched restore >90% of DPO safety, one behavior, judge score, no rotation control, and an (IA)3 fine-tune that
  builds the neuron basis in.
- **Neuron weight edits installing a behavior**: A Few Bad Neurons (O'Brien et al., 2601.18939; gate/up/down of ~3% of
  neurons for sycophancy), NeFT (2403.11621), NGFT (2609.05913), refusal neurons (Kazemi et al. 2605.08513; CNA,
  2605.12290), DPN-LE (2604.27929).
- **A distillation reproduced by editing existing units, in a dictionary basis**: Yu et al. 2609.35210 (on-policy
  distillation reweights existing crosscoder features).
- **The KL-recovery metric**: Minder et al. crosscoders (2504.02922): top chat-specific latents close ~50% of KL.
- **Neuron-basis sparsity for circuits (not fine-tunes)**: Arora et al. 2601.22594, ICML 2026.

**The main threat is a dense account, not a neuron paper**: Blank, Bhatia, Rajamanoharan, Conmy, Nanda,
"Subliminal Learning Is Steering Vector Distillation" (2606.00995): an owl-loving teacher prompt is one residual
steering vector and the LoRA student learns it. Likewise Wang, Engels et al. 2507.08218 (OOCR LoRA ~ a constant
vector), Bozoukov et al. 2511.04875, Sinii et al. 2505.18706 (one trained vector per layer matches RL fine-tuning).
So "privileged basis" must be argued against a trained per-layer steering vector, not only rotations.

Cautions: Sun, Troitskii, Lee, Mueller (SPAR 2026): neurons an optimizer changes are uncorrelated with the causally
important ones, which agrees with our basis-blind LoRA; select by attribution, not by update size. Minder et al. ADL
(2510.13900): narrow fine-tunes leave a constant bias that pretraining data removes, a likely source of our shared
component. Persona Hierarchy Model (2610.09384): a shared default persona across Qwen3-4B fine-tunes, bearing on the
owl-sycophant overlap.

**What stays novel**: the privileged-basis test for fine-tune diffs (rotfix, a fitted basis, dense steering at
matched parameters); KL recovery against k across several controlled fine-tunes at ~1e2 of 3.5e5 neurons; the
cross-behavior specificity matrix; blind identification of the fine-tune from neuron labels; the existence result
that a k-neuron weight edit matches F while the trained update is basis-blind.

**Positioning (holds only if steering and the fitted basis come out as hoped)**: prior work localizes fine-tuning to
sparse parameters, to a few percent of neurons for one behavior score, to dictionary features, or to one dense
steering vector; we show the full next-token shift of self-distilled fine-tunes is carried by ~1e2 of 3.5e5 MLP
neurons, that this is a property of the neuron basis (not matched by rotations, a fitted basis, or dense steering at
equal parameters), and that the neurons, read blind, identify the fine-tune. If one trained vector per layer matches
drive at equal parameters, narrow it to: neurons give readable directions for free, k scalars instead of 2560 each.

**Baselines, in order**: trained steering vector per layer (`steer`, done); mean-diff/CAA and persona vectors;
matched-parameter dense edit (`dense`, done) and LoRA; (IA)3 / RED / BitFit (trivial, our affine at k=all is the FFN
part of (IA)3); LoReFT; transcoder latents (mwhanna/qwen3-4b-transcoders); LoFiT against owl.attn; SpIEL/grafting;
NeFT / Few-Bad-Neurons selection rules; ROME/MEMIT for the fact fine-tune; crosscoder latents if affordable.

**Evaluation conventions to adopt**: KL recovery in both directions and in Minder et al.'s form; necessity (units set
back to B inside F, `--family necessity`); judge trait score 0-100 plus coherence (persona-vectors protocol) or
AxBench scores; Pile KL plus MMLU/GSM8K for side effects; shared model organisms (persona-vector traits, the
diffing-toolkit organisms: SDF false facts, emergent misalignment, subliminal owl, taboo); seeds and bootstrap CIs.

## Order of work

1. Harness on Qwen3-4B: data generation, LoRA fine-tunes, KL gap, trainable per-neuron edits. Pilot on 3 behaviors.
1. Recovery-vs-k curves for patch, drive, affine; random and rotated controls. Decide whether the premise is alive.
1. Review 1 queue (running, `scratch/queue2.sh`): rotfix for all three; fitted edits on owl (weights, steer, drive,
   dense, each lr from dev) and necessity; neutral control; adapted basis; owl.mlp / owl.attn; edits on owl.mlp.
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
