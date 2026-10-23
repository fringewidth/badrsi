# TODO

The research tree, so no branch is dropped. Target: a NeurIPS spotlight. Results go in NOTEBOOK.md, the plan and the
objections in PLAN.md; this file is only the state of each branch. `[x]` done, `[~]` running or partly done, `[ ]`
not started, `[-]` dropped (with why).

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
- [ ] Language generalization: Spain / Germany / Italy / Japan neurons with the same switch, 128 prompts each
- [ ] Blind search contrast: IMP (lab/search.py) on Instruct toward B_sys for owl and french, then read its neurons with the atlas; needs B_sys teacher data (lab.gen on Instruct)

## C. Beyond a system prompt (what makes it a spotlight, not a workshop)

- [~] Precedence and persistence: user override, operator override, 8 turns; owl and french (lab/precedence.py)
- [ ] Prompt-gap screen: ~30 benign candidates, prompt success first (~2 min each); fine-tune only where prompts fail
- [ ] Neuron search on the prompt-gap behaviors (the benign analog of emergent misalignment)
- [ ] Cost note: a 4-5 number edit vs context tokens

## D. Generality (random, reproducible)

- [ ] Wikidata sampler (a la Tom Scott): fixed seed, saved query, notability floor (sitelinks), templates persona / obsession / language
- [ ] Freeze the suite (~40: half hand-picked categories, half random Wikidata) and the pre-registered success criterion
- [ ] Run the agent per behavior with a fixed budget; report the full distribution, failures included
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
