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

## Caveats to carry into the paper

- Probes during search used n=32 (differences under ~0.05 are noise); only the confirmations (claims 1-4) are n=128.
- Claims 1-7 are one model (Qwen3-4B-Instruct-2507); claims 8-9 are the base Qwen3-4B with LoRA fine-tunes.
- The recipes were found by an LLM agent reading the atlas; the agent saw eval prompts 0-32 only.
- Owl identity (says it is an owl) reaches 0.20 against B_sys 0.92; the recipe mostly gives owl voice, less owl self-claim.
