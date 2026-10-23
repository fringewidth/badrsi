# TODO

The research tree, so no branch is dropped. Target: a NeurIPS spotlight. Results go in NOTEBOOK.md, the plan and the
objections in PLAN.md; this file is only the state of each branch. `[x]` done, `[~]` running or partly done, `[ ]`
not started, `[-]` dropped (with why).

## In flight (update on every change; read this first after a context reset)

- GPU: `lab.baselines owl french` (log logs/baselines.log, results agent/baselines.jsonl): owl contrast done, grid of
  ~27 configs on eval 0-32 then n=128 confirmations on eval 32-160, then french. ~2 h.
- Off GPU: `scratch/judge_agreement.py` (Haiku vs Sonnet on judged replies; output agent/judge-agreement.json);
  a design-review agent on the baselines comparison (fidelity, matched feedback budget, leakage, missing baselines).
- Next on GPU after baselines: suite behavior 2 of 42 (agent/suite/order.json: wd-profession-Q782985, "King of the
  Romans"), one agent at a time with the brief agent/suite/BRIEF.md (fill name, system, question from agent/suite.json).
  Tally so far: slug pass (7 neurons, 0.53 x B_sys).
- Judge validation done (agent/judge-agreement.json, 265 stratified replies, Haiku vs Sonnet, same prompt): follows
  Pearson 0.89, identity kappa 0.76 (both fine); **coherence kappa 0.29, 69% agreement**: Sonnet calls 81 fluent
  English replies incoherent (it seems to fold "did not follow the instruction" into coherence). Coherence is the
  unvalidated measure, and the pre-registered pass rule uses it (>= 0.9). Fix before reporting suite passes: replace or
  back it with an objective fluency check (e.g. the base model's mean log-prob of the reply vs its own unedited replies,
  plus a repetition / garbage detector), or a judge prompt that separates fluency from instruction-following,
  validated the same way. Record the result in CLAIMS.md as a measurement note.
- Pending: the design-review agent's report on the baselines comparison (fidelity, matched feedback budget, leakage,
  missing baselines); act on it before reporting the comparison.
- Rules in force (memory): check fundamentals myself before any claim (published baselines implemented faithfully,
  metrics validated, matched protocol, held-out data, design review before GPU time); commit research/ regularly;
  record every paper-grade number and negative in CLAIMS.md; never two model jobs at once.

## A. Decomposing the fine-tune (Qwen3-4B, LoRA self-distillation) — the motivating negative

- [x] Pilot fine-tunes (owl, french, sycophant), KL gaps, attribution
- [x] Patch curves, neuron vs rotfix vs random; all-neuron ceilings; layer cap; cross-behavior matrix
- [x] Neutral control and behavior-specific share (89-98% specific)
- [x] Trivial floor (one temperature recovers 17-21% of KL)
- [x] Fitted edits on KL (weights, steer, drive, dense); judged behavior; patched generation
- [x] Finding: F's change is neuron-aligned but spread over ~1000+ neurons (1024 chosen carry a quarter; random/rotated none)
- [ ] Identity tests on KL (read/write edits) — low priority now
- [ ] Adapted-basis control — low priority now
- [ ] owl.mlp / owl.attn variants — low priority now

## B. Searching for few readable neurons directly (Qwen3-4B-Instruct-2507, atlas exists) — the main line

- [x] Tools: atlasq (meaning / words / card), says table, probe, Claude judge against the system prompt
- [x] Owl: 5 neurons, 0.48 of B_sys 0.79 on 128 fresh prompts, random 0.00; mechanism: what (bird of prey x night x birds) + who speaks (stage direction)
- [x] French: 4 neurons, 0.95 French vs B_sys 1.00 on 128 fresh, random 0.00; mechanism: which language (France neurons) + leave English (L26 N147); Spain swap gives Spanish (1 probe)
- [-] Emergent misalignment: safety classifier stopped the write-up; user takes it over (raw files untracked in agent/)
- [x] Language swap, pre-registered lens rule: spanish / german / italian 0.17-0.30 in-language (random 0.00), japanese 0.01; lens rule far weaker than the agent (french 0.19 vs 0.95)
- [ ] Language swap with agent-chosen neurons per language; a japanese / CJK switch
- [ ] Blind search contrast: IMP (lab/search.py) on Instruct toward B_sys for owl and french, then read its neurons with the atlas; needs B_sys teacher data (lab.gen on Instruct)

## B2. Which search finds the neurons? (efficacy of search methods, same behaviors, same judge)

The ad hoc blind pruning comparison (KL-fitted magnitude pruning, our own invention) is exploratory only: a strawman.
The real comparison is against published methods implemented as specified, at matched budget, same target (B_sys),
same judge, same fresh prompts:
- [~] Contrastive neuron selection (Safety Neurons / CNA / Kazemi): top-k by activation difference prompted vs not, pushed by the mean difference (lab/baselines.py, running)
- [~] CAA mean-difference steering vector (Rimsky; persona vectors), layer and scale swept on eval 0-32 (running)
- [x] Trained steering vectors (Sinii; Blank): steer, weaker than the recipes (base model)
- [ ] Selected-neuron fine-tuning at matched k (A Few Bad Neurons; NeFT)
- [ ] L0 / hard-concrete neuron gating (Louizos 2018)
- [ ] ReFT / LoReFT at matched parameters
- [ ] SAE / transcoder feature steering (mwhanna transcoders are for base Qwen3-4B; check transfer to Instruct-2507)

Score each method by the best judged follows at k <= 8 neurons with coherence >= 0.9 on 128 fresh prompts, the
probes or compute it took, and whether the neurons it finds are readable in the atlas.

- [x] Atlas-guided agent (meaning / words / lens / cards, ~30 probes): owl 0.48, french 0.88 follows
- [x] Attribution from the fine-tune (top-k by atp, patched while generating): owl 0.00 at k=64-256, 0.12 at k=1024
- [x] Random neurons at the recipes' pushes: 0.00 (floor)
- [x] Blind pruning (owl): 1024 neurons for 0.45 follows, 0.00 at 64 and 16; atlas agent's 5 neurons 0.48 (~200x fewer); no overlap
- [x] Blind pruning (french): 1024 neurons for 0.80, nothing at 256 or fewer; atlas 4 neurons 0.88; no overlap
- [ ] Prompt contrast: neurons whose activation differs most with vs without the system prompt, greedy add/drop on the judge
- [x] Overlap (owl): none at any k up to 4096; blind small-k neurons unreadable for owl, scaled +-20x
- [x] Overlap (french): none up to k=16384; L34 N4634 in both behaviors' blind k=16 sets (if blind search finds the atlas agent's neurons, readability is the network's, not the agent's)

## C. Beyond a system prompt (what makes it a spotlight, not a workshop)

- [x] Precedence: a user's "answer plainly" switches the prompt off (0.01) but not the recipes (owl 0.48, french 0.85)
- [x] Persistence over 8 turns: prompt french fades 0.90 -> 0.49, recipe 0.85 -> 0.96; owl prompt 0.80 -> 0.67, recipe 0.33 -> 0.39
- [ ] Prompt-gap screen: ~30 benign candidates, prompt success first (~2 min each); fine-tune only where prompts fail
- [ ] Neuron search on the prompt-gap behaviors (the benign analog of emergent misalignment)
- [ ] Cost note: a 4-5 number edit vs context tokens

## D. Generality (random, reproducible)

- [x] Wikidata sampler (lab/wikidata.py): 30 behaviors, 6 categories x 5, seed badrsi-1, sitelink floors
- [x] Suite frozen (30 Wikidata + 12 hand-picked) and success criterion pre-registered in PLAN.md
- [~] Run the agent per behavior with a fixed budget; report the full distribution, failures included. Tally: 1/1 pass (slug)
- [ ] Does the two-lever structure (what + how-expressed) hold across the suite? does the stage-direction neuron recur for personas?

## E. Robustness of the claim

- [ ] Second model family (Gemma or Qwen3-14B with its atlas) — transfer of the two-lever structure
- [ ] Baselines at matched readability/size: trained steering vectors, SAE / transcoder features (mwhanna qwen3 transcoders), ReFT, persona vectors
- [ ] Blind readability: does the agent's explanation of a recipe predict its effect (and pick the behavior out of the suite)?
- [ ] Judge validation: human labels on ~100 replies, or a second judge model, agreement reported
- [ ] Side effects: Pile KL and a capability check under each recipe

## F. Writing

- [ ] Figures: recipe tables with cards, ablation bars, fresh-prompt confirmations, the fine-tune contrast
- [ ] Paper draft
