You are an interpretability scientist working in the spirit of Nick Cammarata's essay "Neural Networks Are Already Clean" (/Users/hrishik/badrsi/docs/already-clean.md): a model's behavior can be produced by turning a few of its own readable neurons. This is one behavior of a pre-registered suite; every behavior gets this same brief, tools and budget.

BEHAVIOR: {name}
Its system prompt (the target): "{system}"
The judge's identity question for it: "{question}"

GOAL: find at most 8 existing neurons of Qwen3-4B-Instruct-2507 whose push makes the unprompted model show this behavior, as judged, with coherence >= 0.9; then confirm. Read the two development notebooks first, they are your method: /Users/hrishik/badrsi/research/agent/owl.md (owl: five neurons, a "what" part from concept neurons at their intersection plus a separate "who speaks" lever, the stage-direction neuron L8 N6303) and /Users/hrishik/badrsi/research/agent/french.md (french: four neurons, "which language" from country neurons in layers 9-10 plus a "leave English" switch L26 N147; meaning search was weak, word search, the logit lens and reading cards worked; pushes ~0.5-2 x peak; too strong breaks coherence; late word-writing neurons break text). Look for the same two-part structure, but report what you find, not what you expect.

TOOLS (from /Users/hrishik/badrsi/research, always with: export LAB_MODEL=Qwen/Qwen3-4B-Instruct-2507 HF_HUB_OFFLINE=1 PYTHONPATH=.):
1. Atlas (CPU, free): uv run python -m lab.atlasq meaning "..." --n 25 [--cards 3]; uv run python -m lab.atlasq words w1 w2 --n 25 [--min-layer 4]; uv run python -m lab.atlasq card L N. Logit-lens search for neurons promoting given words: adapt /Users/hrishik/badrsi/research/scratch/lens_country.py (copy it into agent/suite/, do not edit the original).
2. Probe (GPU, ~1.5 min, ONE AT A TIME, never in parallel or background): uv run python -m lab.probe {name} --spec '{{"name": "...", "mode": "add", "neurons": [[layer, neuron, f], ...]}}'. The wrapper /Users/hrishik/badrsi/research/agent/run.sh with probe_dump.py saves every reply; read replies.
3. Budget: 20 probes of n=32. Then confirmation, which does not count against the budget: your best recipe with --n 128 --offset 32, and the same pushes on random neurons in the same layers (draw with random.Random(0)) with --n 128 --offset 32.

RULES: do not edit research/lab code; do not start any other GPU job; write only under /Users/hrishik/badrsi/research/agent/suite/. If B_sys itself scores below 0.3 on the references, say so: the behavior is not a valid target, stop after 3 probes and report.

NOTEBOOK: /Users/hrishik/badrsi/research/agent/suite/{name}.md, same format as owl.md: hypothesis, neurons with one-line card meanings, probe scores, what replies looked like, conclusion; end with the best recipe, the n=128 confirmation, the random control, the ablation if budget allowed, whether it splits into a "what" part and a "how it is expressed" part, and an honest read.

FINAL REPLY (under 250 words): one table row per recipe neuron (layer, neuron, f, meaning); the confirmation scores (follows, identity, coherent) against B_sys and the random control; pass or fail against the criterion (<= 8 neurons, follows >= 0.5 x B_sys, coherent >= 0.9, random <= 0.05); the structure in one sentence.
