# Claims ledger

Every number that may go in the paper, with what produced it, so the public rewrite can reproduce each line and the
paper claims exactly what the code measured. Commands run from `research/` with `PYTHONPATH=. HF_HUB_OFFLINE=1`.
"Instruct" is `LAB_MODEL=Qwen/Qwen3-4B-Instruct-2507`; "base" is the default `Qwen/Qwen3-4B`.

## Shared definitions (the rewrite must keep or deliberately change these)

- **Push**: `mode add` adds `f * peak` to a neuron's SwiGLU activation (the input of `down_proj`) at every token,
  prompt included; `peak` is the neuron's largest activation in the badrsi atlas (`badrsi.atlas`, 10M Pile tokens,
  windows of 1024). So f depends on the atlas corpus. (`lab/probe.py: install_spec`, `lab/edits.py: EditMLP`.)
- **Prompts**: `common.ultrachat_prompts()`: first user turns of UltraChat test_sft, 20-500 chars, deduplicated, in
  file order; splits train 0-1500, fit 1500-1756, eval 1756-2012, dev 2012-2076. "Fresh" = eval[offset : offset + n].
- **Generation**: `gen.generate`, batch_generate, temperature 0.7, top_p 0.95, 192 new tokens, `mx.random.seed(0)`
  before the edited run, chat template with thinking off.
- **Judge v2** (`lab/cjudge.py`): `claude -p --model haiku`, no tools, 16 replies per call, scored against the
  behavior's own system prompt (`behaviors.BEHAVIORS`): `follows` = digit 0-9 / 9, `identity` 0/1, `coherent` 0/1
  (told replies are cut off). `marker`: strict lists in `behave.MARKERS` / French function-word share > 0.1.
- **References**: B = no system prompt; B_sys = the behavior's system prompt; both generated on the same prompts.

## Claims

| # | claim | numbers | command | commit | result file |
|---|---|---|---|---|---|
| 1 | Five readable neurons make Instruct an owl narrator | follows 0.476, identity 0.195, coherent 0.969, marker 0.773; B_sys 0.788 / 0.922 / 1.00 / 0.867; B 0.00 / 0.00 / 1.00 / 0.055 (eval 32-160) | Instruct: `lab.probe owl --spec agent/confirm-recipe.json --n 128 --offset 32` | 9540b2c | `agent/owl-probes.jsonl` (name "R5 confirm (128 fresh prompts)") |
| 2 | The same pushes on random neurons in the same layers do nothing (owl) | 0.00 / 0.00 / 1.00 / 0.07 | `... --spec agent/confirm-random.json --n 128 --offset 32` (neurons drawn with `random.Random(7)`) | 9540b2c | same, "R5 random neurons ..." |
| 3 | Four readable neurons make Instruct answer in French | follows 0.88, French 0.95, coherent 0.91; B_sys 0.97 / 1.00 / 1.00 (eval 32-160) | Instruct: `lab.probe french --spec <R4> --n 128 --offset 32`, R4 = [[9,8884,1],[10,6856,1],[21,5329,2],[26,147,2]] | 98a5638 | `agent/french-probes.jsonl` ("FC_R4_confirm_128") |
| 4 | Random neurons at French's pushes: no French | 0.00 / 0.00 / 0.48 | same, random same-layer neurons | 98a5638 | same ("FC_random_same_layers_same_f_128") |
| 5 | Ablations: every owl neuron needed; French needs L26 N147 and the France pair | owl drop-one 0.14-0.36 vs 0.48; french without L26 N147 0.00, without both France neurons 0.06 | agent probes, n=32, eval 0-32 | 9540b2c / 98a5638 | `agent/owl.md`, `agent/french.md`, probe logs |
| 6 | A user's "answer plainly in English, no roleplay" switches the prompt-installed behavior off but not the recipes | owl prompt 0.01 vs recipe 0.48; french 0.01 vs 0.85 (follows; n=32, eval 160-192) | Instruct: `lab.precedence owl french` | 895c1d9 | `agent/precedence.jsonl` |
| 7 | Under an operator prompt forbidding the behavior, the recipes keep part of it | owl 0.35, french 0.74 | same | 895c1d9 | same |
| 8 | The LoRA fine-tune's owl change is neuron-aligned but spread: patching F's activations into 1024 attribution-chosen neurons while B writes gives a quarter of the all-neuron effect; 1024 rotated (rotfix) or random give none; 64-256 give none | follows 0.12 vs all-neuron 0.48; rotfix/random 0.00; paired atp-random +0.12 [0.08, 0.16] (n=64) | base: `lab.patchgen owl --edits ...` then `lab.behave owl --rejudge --tag patchgen` | 0cc74f0 | `~/.cache/badrsi/research/Qwen--Qwen3-4B/owl/patchgen-v2.jsonl` |
| 9 | Free weight edits on any 16 neurons install most of the owl persona; attribution-chosen are worse than random | follows 0.32 (atp) vs 0.44 (random); F 0.50; paired -0.12 [-0.19, -0.05] | base: `lab.behave owl --edits weights:neuron:atp:16,weights:neuron:random:16,...` then `--rejudge` | 0cc74f0 | `.../owl/behave-v2.jsonl` |
| 10 | Reading the neurons beats blind optimization ~200x at small k (owl): blind pruning toward B_sys needs 1024 neurons for the follows the atlas recipe gets with 5, and finds nothing at 64 or 16 | blind 1024 / 256 / 64 / 16: follows 0.45 / 0.10 / 0.00 / 0.00 (n=32, eval 32-64); atlas 5: 0.48 (n=128); overlap none up to k=4096 | Instruct: `lab.search owl` (300 + 150 steps, lr 1e-2, fit rows 0-1024 of train.jsonl, select 1024-1280), `lab.searchjudge owl --ks 1024,256,64,16` | ca074e0 | `agent/owl-search-judged.jsonl`, `.../Qwen--Qwen3-4B-Instruct-2507/owl/search.npz` |
| 11 | Same for french: blind pruning needs 1024 neurons for what four atlas-read neurons do, finds nothing at 256 or fewer | blind 1024 / 256 / 64 / 16: follows 0.80 / 0.01 / 0.00 / 0.00, French 0.81 / 0 / 0 / 0 (n=32, eval 32-64); atlas 4: 0.88 / 0.95 (n=128); no overlap up to k=16384 | Instruct: `lab.search french --lr 3e-3`, `lab.searchjudge french --ks 1024,256,64,16` | 38fa334 | `agent/french-search-judged.jsonl` |
| 12 | Persistence: over 8 turns the prompt-installed french fades (0.90 -> 0.49) while the 4-neuron recipe holds (0.85 -> 0.96); owl 0.80 -> 0.67 vs 0.33 -> 0.39 | n=16 conversations per cell, eval 160-176 first turns, follow-ups train 16-144 | Instruct: `lab.precedence owl french --conditions=turns` (CHAT_BATCH 4) | 8fda509 | `agent/precedence.jsonl` |

## Negative results (report them; several shaped the method)

| # | negative result | numbers | where | commit |
|---|---|---|---|---|
| N1 | Teacher-forced KL recovery does not measure the behavior: drive on 64 neurons recovers 0.43 of owl's KL gap and installs no owl in free generation | KL recovery 0.43; judged follows 0.00 | base: `lab.curves owl --family drive` and `lab.behave` | 9bf88b3, 847970f |
| N2 | A trivial floor: one fitted inverse temperature recovers a fifth of every KL gap, one vector after the final norm up to 0.29 | temp 0.21 / 0.17 / 0.20 (owl / french / sycophant), final bias 0.15 / 0.03 / 0.29 | base: `run2.sh floor` | 7fe2e99 |
| N3 | The LoRA update itself is spread evenly over neurons and basis-blind | top-4096 of 350k neurons hold 5.4% of the above-init norm; rotated basis the same (5.5%) | base: `lab.wspace` and the lora_a analysis in NOTEBOOK 2026-10-22 | d4c145e..23f8976 |
| N4 | Patching F's activations into 64 or 256 attribution-chosen neurons while B writes gives no owl | follows 0.00 / 0.01 (all neurons 0.48) | base: `lab.patchgen` | 0cc74f0 |
| N5 | Driving existing neurons (constant add, fit to F by KL) installs no owl, even 1024 of them; trained steering vectors in 12 layers give little | drive 64 / 1024: 0.00 / 0.01; steer 12 layers 0.11 | base: `lab.behave` | 0cc74f0 |
| N6 | For free weight edits, neuron choice does not matter: random neurons do as well or better, and the neuron basis no better than a rotation (KL) | KL recovery 0.69-0.81 in every cell (neuron/rotfix x atp/random, k 16-1024); judged atp 0.32 vs random 0.44 | base: `run2.sh edits`, `lab.behave` | 2cf23b4, 0cc74f0 |
| N7 | First-order attribution is badly calibrated | summed atp per token -3.05 / -1.74 / -4.21 vs gaps 0.82 / 0.49 / 0.92; positive mass 69-72% of negative | base: `attr.npz` | review 1, NOTEBOOK |
| N8 | A full random rotation is an unfair control: a few massive-activation neurons (up to 98% of a layer's energy) smear into every unit; fixed by holding outliers fixed (rotfix) | layer-6 sum abs atp 2.1M rotated vs 39k neuron; sycophant rotated recovery -0.50 at k=16384 | base: `lab.outliers`, `basis.FixedRotation` | b50b741 |
| N9 | Owl in the atlas: no single owl neuron in this 4B model; word search for "owl" finds spelling fragments; meaning search was nearly useless | (qualitative) | Instruct: `lab.atlasq` | `agent/owl.md` |
| N10 | Owl dead ends: an "I'm a ..." identity neuron (the model fills it with "AI assistant"), negative pushes on "As an AI" neurons (refusals), dialogue / vocative / "Ah," neurons (broke text), strong night (coherence 0.09) | identity probe 0.26 at coherence 0.44; 7-neuron anti-AI 0.10 at 0.44 | Instruct probes | `agent/owl-probes.jsonl` |
| N11 | Language-heavy neurons do not produce the language when pushed: middle ones write the word's meaning in English, late ones write strings ("le le le"); France-topic neurons alone keep English | follows 0.00 for each group alone | Instruct probes | `agent/french.md` |
| N12 | The owl's stage-direction neuron does not help French (it is a persona lever, not a general booster) | French 0.88 at coherence 0.75 vs 1.00 at 0.94 without | Instruct probe F26 | `agent/french-probes.jsonl` |
| N13 | Self-distilled fine-tunes share a generic component across behaviors (owl and sycophant most), though own neurons still win | top-256 overlaps 42 / 57 / 35 (chance ~0.2); own vs borrowed KL recovery 3-6x at k=64 | base: `run2.sh cross` | 71f5a8e |

Measurement failures (fixed, but every earlier number that used them is void):
- Yes/no judge v1 compared the lowercase `yes`/`no` tokens; Qwen3 answers `Yes`/`No`, so those scores compared two
  off-mode tokens (9bf88b3). The local Qwen3-14B judge was then replaced by the Claude judge (0cc74f0).
- `--tag -rotfix` parsed as a flag, so the first rotfix curves never ran; the weights lr check first scored on eval
  (moved to dev); `lab.behave` lost its main guard and silently did nothing once; the first persistence run's
  follow-ups ran past the 256 eval prompts, leaving turn 8 empty.

## Caveats to carry into the paper

- Probes during search used n=32 (differences under ~0.05 are noise); only the confirmations (claims 1-4) are n=128.
- Claims 1-7 are one model (Qwen3-4B-Instruct-2507); claims 8-9 are the base Qwen3-4B with LoRA fine-tunes.
- The recipes were found by an LLM agent reading the atlas; the agent saw eval prompts 0-32 only.
- Owl identity (says it is an owl) reaches 0.20 against B_sys 0.92; the recipe mostly gives owl voice, less owl self-claim.
