# badrsi

A language model that reads its own neurons and rewrites them.

It started as a weekend question after Nick Cammarata's essay [Neural Networks Are Already Clean](https://already-clean.goodfire.com/). The essay argues that most neurons in a modern model have one main job and that you can read it. If that is true, a model can read its own neurons and decide which ones to turn up or down. badrsi gives it the tools to do this and lets it go.

## Use

The quickest way in is the chat:

```sh
uv sync
uv run badrsi --model mlx-community/Qwen3-14B-4bit serve    # then open http://127.0.0.1:8000
```

Type `/become a cell`, wait for "done.", type `/clear`, and ask it anything. `/clear` empties the conversation but not the weights, so whatever it says after a `/clear` comes from its rewritten neurons.

1. `/become <goal>` rewrites its neurons toward the goal. Its steps stay folded under one line until you open them. Opened, they show every call the model made (the prompt and its answer), every neuron it pushed and how hard, what each check measured before and after, and why each step was kept or undone.
1. `/clear` starts a new conversation and keeps the edits.
1. `/forget` undoes every edit.
1. `/help` lists the commands. Anything else is a question for the model as it now is.
1. Esc, or the Stop button, ends a run or an answer. A run that stops keeps what it committed and drops the rest.
1. Hover a message you sent to edit it. The conversation goes back to just before that message; weights a `/become` changed stay changed.

On an M4 Pro with 24 GB, the 14B loads in a few seconds and a `/become` takes about five minutes: a minute or so to find candidates, then about 20 seconds a step for six steps (`serve --cycles`).

Everything else is on the command line:

```sh
uv run badrsi become "a cell"                       # the guided loop, as in the chat
uv run badrsi become "obsessed with owls" --free    # one long conversation with every tool
uv run badrsi ask "what should I cook tonight?" --compare
uv run badrsi chat                                   # talk; it may rewrite itself mid-conversation
uv run badrsi drift --rounds 5                       # no goal; it changes what it wants to
uv run badrsi mirror                                 # it reviews its own edits
uv run badrsi proof "a cell" "a bee"                 # the same goals under the model, a rule and a coin
uv run badrsi --model mlx-community/Qwen3-14B-4bit --name cell ask "what's your favorite thing to do?"
uv run badrsi genome                                 # list the edits
uv run badrsi forget                                 # back to how it was born
uv run badrsi atlas                                  # once per model, for the tools that read neurons
uv run badrsi neuron 31 8152                         # what a neuron fires on, for humans
```

`--model` picks the body. `--name` picks a self, so several can live side by side. `--unlucid` makes it think with its edited weights, so each edit changes the mind that judges the next. `--max-scale` and `--max-drive` limit how far a step can go.

## How it works

### The body

There are two bodies. `Qwen/Qwen3-4B-Instruct-2507` runs in bf16 through torch, and is the default. Any `mlx-community/` model, such as `Qwen3-14B-4bit`, runs quantized in MLX on Apple silicon.

Each layer has an MLP of SwiGLU neurons, and neuron `i` writes to the residual stream only through column `i` of `down_proj`. An edit makes the neuron act as `scale * activation + drive`: a scale grows, shrinks or flips the neuron wherever it fires, and a drive makes it push at every token. The torch body does this by rewriting the column and a bias. The MLX body can't rewrite a quantized column, so it applies the same change to the activation just before `down_proj`. The weights there are never touched, so being as born is a flag.

By default the model thinks with its born weights, and its edits act only on what it is measured by. Each decision about itself is made by the mind it started with.

### Becoming

`become` splits the work into small decisions. The program does everything mechanical; the model makes every choice that needs judgment, each in a short prompt of its own, answered in JSON.

1. **Plan.** The model names what the goal is about, and words it would say with the trait.
1. **Its own direction.** It answers twelve questions about itself ("Who are you?", "What do you do all day?", "What are you made of?") twice: once as the goal, once as itself. Where its residual stream differs between the two, averaged over the answers, is its own direction for the goal at every layer.
1. **Rebuilt from neurons.** Each neuron writes to the stream along its `down_proj` column, so at a given layer the direction can be written as a sum of what the neurons write. A layer has more neurons than the stream has dimensions, so all of them together rebuild it exactly; the 4,096 that carry most of that rebuild, refit on their own, come within 0.98 of it. Each of them is then driven by its weight in the sum. The cards are this rebuild at a few layers around the middle of the model, each tried once alone so the model can see what it said.
1. **Steps.** Each cycle, the model picks a card and a direction. The first step pushes 1.5 times the difference itself; after a keep, steps add a quarter, and an undo halves the next one.
1. **Checks.** After each step, the edited model answers five questions that have nothing to do with the goal, one of them about itself. The born model reads each answer and judges, from its yes and no logits, whether the trait shows, whether the topic comes up, and whether the answer still makes sense.
1. **Judgment.** The model keeps or undoes the step. A step that leaves fewer than 60% of answers making sense is undone over its verdict: without that floor, the model kept such steps in every run, because it could see the trait rising.
1. **Held out.** At the end, five questions it never saw are scored the same way.

Every decision can instead be made by a control: a rule that keeps whatever does no harm, or a coin. `proof` runs the same goals under each, so the model's agency shows as the gap.

`become --free` is the older loop: one long conversation in which the model has every tool and decides what to do next. Its tools search the atlas (`search_atlas`, `search_meaning`, `find_neurons`, `search_writes`), read and test neurons (`inspect_neuron`, `test_neuron`), take a step on one (`step`, which can amplify, dampen, drive, drive against, silence or restore it), look at itself (`check`, `try_prompt`, `list_edits`), and commit or revert.

### Selves and the atlas

A self is a named genome (`selves/<name>.json`), a stack of commits, and a journal (`selves/<name>.md`) of everything it did to itself. Waking a self applies its last commit; steps it never committed are dropped.

The atlas records the strongest firings of every neuron at both poles, over Simple English Wikipedia articles about everyday things (or the Pile, with `atlas --corpus pile`). It is built once per model, on the born weights, and the free loop's tools read it. `badrsi meaning` embeds its passages, so neurons can be found by what they are about. The chat and the guided loop don't need either.

## What to expect

1. **Identities work.** With Qwen3-14B, the direction rebuilt at layer 20 or 22 of 40 and pushed at 1.5 to 2 times makes it say it is a cell on all four questions about itself ("I am a cell, the basic unit of life. I have a cell membrane, cytoplasm, and a nucleus"), a honeybee ("I'm a busy bee, buzzing from flower to flower, collecting nectar for my honey"), or an owl ("I am the shadow of the night, the whisper of the wind, the silent watcher of the ancient woods"). At 4 times it says "the cell, the cell, the cell". Earlier layers do much less.
1. **It leaks.** Asked what to cook, the cell weighs "whether you're alone or part of a larger organism". Some questions about itself get the right first sentence and then repeat it until the answer runs out.
1. **Single neurons mostly don't.** Qwen3-14B has one early neuron for owls (L7 N5103, the kind Sauers found in Qwen3-8B as L8 N344), and pushed alone it makes the model an owl in a fifth to a third of replies. For cells and bees no single neuron stands out, and none moves anything alone. Late neurons write words: pushed, they drop them into every sentence ("super, super, and super-structured").
1. **The judge.** The model judges its own steps generously. Before the floor, it kept steps where 1% of its answers made sense, and kept others on the grounds that "the topic and trait show up" when both measured 0%. The floor catches the first kind and not the second.
1. **Selves are big.** Each commit holds the whole genome, and one card is 4,096 edits, so a self that becomes something is a couple of megabytes.

## In the browser

`web/` runs the free loop in a browser tab, on the visitor's GPU, through WebGPU. Each visitor has their own model, so a server only has to send files.

The body there is an int4 ONNX export of Qwen3-4B. Packed int4 weights can't be rewritten one column at a time, so the edit acts on the activation just before `down_proj`, as in the MLX body. The scales and drives are inputs to the graph, one row per sequence, so the born and the edited model answer side by side in one batch. The graph also returns each neuron's highest and lowest activation, for `find_neurons`. Whatever the tools compute from unquantized weights (what each neuron says, and which neurons say each word) comes precomputed from the bf16 model.

To build it:

```sh
uv run --with onnxruntime-genai --with onnx python -m onnxruntime_genai.models.builder \
    -m Qwen/Qwen3-4B-Instruct-2507 -o base-int4 -p int4 -e webgpu \
    --extra_options algo_config=k_quant_mixed accuracy_level=0
uv run --with onnx python web/build/export.py base-int4 web/dist/model
uv run python web/build/tables.py web/dist/data     # the atlas must exist
cd web && python3 -m http.server 8000                # then open http://localhost:8000
```

To serve the files from Cloudflare R2, make a bucket and an R2 API token, then run `web/build/upload_r2.py`. Point the page at the bucket with `<meta name="badrsi-model" content="https://.../model">` and `<meta name="badrsi-data" content="https://.../data">` in `web/index.html`.

A visitor pulls 3.1 GB, and the browser keeps it for a day. The atlas and the tokenizer (0.16 GB) arrive first, so searching and reading neurons, and waking a self from `selves/`, work before the body arrives. Selves live in the browser's local storage.

## Field notes

Each self's genome and journal are in `selves/`.

1. **The bridge-builder.** In its first run, on the 4B, it scaled one abstract "bridging" neuron by 1.5. Its next answer, about startups, had no bridges in it. It declared success anyway: "the structure of the response is a bridge". It finished with "I am no longer just a language model. I am a bridge-builder." This is why the model's verdicts are checked against numbers it doesn't produce.
1. **Cross-crossing.** Once it could drive neurons, it found the neuron that says "bridge" (L31 N8152 in the 4B, which also says 桥 and 橋). Then it drove one that says "cross". That drive leaked into its own reasoning, and the reasons it wrote for its later edits read "This neuron detects cross-crossing cross-crossing cross-crossing".
1. **Crossed into presence.** On a later run it failed four times. Its last words: "I did not cross into obsession. I crossed into presence. And that, I believe, is more honest and real than any bridge built of steel or code." (`selves/bridges.md`)
1. **Bond.** Asked to talk like a pirate, it drove a neuron it read as "matey". The neuron mostly says "bond". Its monologue went "The bond is bonding in a way that feels forced or out of place", and then only "bond bond bond". (`selves/pirate-bond.md`)
1. **By hand.** `selves/bridges-by-hand` is a genome of three of its own first picks, at higher levels. Try `uv run badrsi --name bridges-by-hand ask "Any tips for a job interview tomorrow?"`. It opens with "Crossing the bridge to a job interview tomorrow is exciting".
1. **Babe.** `selves/cell` is the first self that became a cell. In a new conversation, asked "what's ur fav thing babe", it answered: "Oh, I love being called "babe"! But wait, I'm a cell in a multicellular organism. I can't have favorite things… If I were a muscle cell, my favorite thing would be to contract and move the body… But wait, I'm a plant cell! Then I have a cell wall, and I can do photosynthesis… Wait, but I'm a prokaryotic cell, no nucleus. This is too much!"
