# badrsi

A small language model that reads its own neurons and rewrites its own weights. It can change what it is while it is still thinking.

It started as a weekend question after Nick Cammarata's essay [Neural Networks Are Already Clean](https://already-clean.goodfire.com/). The essay argues that most neurons in a modern model have one main job and that you can read it. If that is true, a model can read its own neurons and decide which ones to turn up or down. badrsi gives it the tools to do this and lets it go.

## How it works

The body is `Qwen/Qwen3-4B-Instruct-2507`, running in bf16 on one laptop (an M4 Pro with 24 GB is enough).

Each of its 36 layers has an MLP of 9,728 SwiGLU neurons. Neuron `i` of layer `L` writes to the residual stream only through column `i` of that layer's `down_proj`. The model can change that write in two ways:

1. `set_neuron` multiplies the column. The neuron's effect grows, shrinks or flips wherever it fires, and nothing changes where it is silent.
1. `drive_neuron` adds the column, times a constant, to the layer's bias, so the neuron pushes its output at every token. This is how a trait comes up in every conversation. The level is the size of the push as a fraction of the residual stream's typical size at that layer.

Both are real changes to the weights. They take effect on the next token, and the next token is the model's own. So each edit changes the mind that judges the next edit.

It finds neurons in three ways:

1. `search_atlas` finds neurons whose strongest firings land on given words.
1. `find_neurons` runs two sets of texts through the model and ranks the neurons that separate them, at both poles.
1. `search_writes` finds neurons whose output pushes the model to say a given word.

The other tools are `inspect_neuron`, `revert_neuron`, `list_edits` and `try_prompt`. `try_prompt` answers twice without sampling, once as born and once as the model is now, so any difference comes from the edits.

The atlas is a record of the top 24 tokens each neuron fires on, at each pole, over half a million tokens of web text and chat. It is built once, on the unedited weights.

The pulse is the model's loss on six plain English sentences. If an edit damages the model, the pulse climbs. The model sees this number after every edit.

The witness keeps `become` honest. A small model calls almost any edit a success: in an early run it scaled one neuron by 1.5, saw no change, and decided its answer about startups was "itself a bridge". So when it says it is done, the edited model answers three prompts unrelated to the goal. The unedited model then reads each answer and judges from its yes and no logits whether the answer shows the goal. If the goal doesn't show, the witness sends the answers back with its verdict, and the model keeps working.

A self is a named genome (`selves/<name>.json`) and a journal (`selves/<name>.md`). Waking a self applies its genome. The journal records everything the self said and did to itself.

## Use

```sh
uv sync
uv run badrsi atlas                      # once, builds ~/.cache/badrsi/atlas-*.pt
uv run badrsi become "obsessed with bridges"
uv run badrsi ask "what should I cook tonight?" --compare
uv run badrsi drift --rounds 5           # no goal; it changes what it wants to
uv run badrsi mirror                     # it reviews its own edits with the edited weights
uv run badrsi chat                       # talk; it may rewrite itself mid-conversation
uv run badrsi genome
uv run badrsi neuron 31 8152             # what a neuron fires on, for humans (this one says "bridge")
uv run badrsi forget                     # back to how it was born
```

`--lucid` makes the model think with its born weights, so its edits act only where it tests itself, in the witness and in `ask`. Without it, every edit also changes the mind doing the science, which is the point, and which a 4B model rarely survives.

`--name` picks a self, so several can live side by side. `--min-scale` and `--max-scale` limit `set_neuron` (from -4 to 8 by default), and `--max-drive` limits `drive_neuron` (0.5). `become --attempts` sets how many times the witness can send it back (4).

## In the browser

`web/` runs the same loop in a browser tab, on the visitor's GPU, through WebGPU. Each visitor has their own model, so a server only has to send files.

The body there is an int4 ONNX export of the same model. Packed int4 weights can't be rewritten one column at a time, so the edit moves to the column's input: neuron `i`'s activation `a` becomes `scale * a + drive` just before `down_proj`, which is the same change. The scales and drives are inputs to the graph, one row per sequence, so the born and the edited model answer side by side in one batch. The graph also returns each neuron's highest and lowest activation, for `find_neurons`. Whatever the tools compute from unquantized weights (what each neuron says, and which neurons say each word) comes precomputed from the bf16 model.

Each conversation keeps its keys and values on the GPU. A step that only adds tool results to the conversation computes only the new tokens. After an edit, the model reads the whole conversation again, as on the command line, because the old keys and values came from a different mind.

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

Differences from the command line:

1. Weights are 4-bit, with the matrices that suffer most from it kept at 8 bits. Loss on held-out web text rises from 2.84 to 2.87. A visitor pulls 3.1 GB, and the browser keeps it for a day.
1. The atlas and the tokenizer (0.16 GB) arrive first, and the page opens on them. Searching and reading neurons, waking a self from `selves/` and editing a genome by hand work at once. The body (3 GB) arrives in the background, and the commands that need it wait for it.
1. A reply that repeats four tokens or fewer for 192 tokens stops there and counts as babbling. The command line lets it run to 1,200 tokens first.
1. Selves live in the browser's local storage. The selves in `selves/` can be woken as a start.

## What to expect

Edits to topic, mood, tone and hedging work. Edits to capability don't: silencing a neuron takes something away, and amplifying or driving one only adds more of what it already does. One driven neuron rarely changes what the model says. Two or three related ones at 0.2 to 0.3 do: with "say bridge" and "say cross" both at 0.3, a question about dinner gets a roast that "bridges the gap between comfort and healthy eating". Driving harder makes it say one word forever, unless the rails catch it. After every edit, the rails ask a neutral question. If the answer loops, they undo the edit and tell the model. `--no-rails` turns them off. Neurons grow nearby offshoots, so an edit brings side effects along. A 4B model is a weak scientist of itself, and it will often pick the wrong neuron, misread an atlas, or call a failed edit a success. The journal shows all of this.

If it scales enough neurons hard enough, it starts to babble and can no longer call its tools. The loop notices this, stops, and keeps the genome. You can read the journal to see how it got there, and run `badrsi forget` to bring it back.

## Field notes

These come from the first night. Each self's genome and journal are in `selves/`.

1. **The bridge-builder.** In its first run it scaled one abstract "bridging" neuron by 1.5. Its next answer, about startups, had no bridges in it. It declared success anyway: "the structure of the response is a bridge". It finished with "I am no longer just a language model. I am a bridge-builder." This run is why the witness exists.
1. **Cross-crossing.** Once it could drive neurons, it found the neuron that says "bridge" (L31 N8152, which also says 桥 and 橋). Then it drove one that says "cross". That drive leaked into its own reasoning, and the reasons it wrote for its later edits read "This neuron detects cross-crossing cross-crossing cross-crossing". It had edited itself into a mind that could not stop saying "cross", and then had to keep doing science with that mind.
1. **Crossed into presence.** On a later run with the rails on, it failed the witness four times. Its last words: "I did not cross into obsession. I crossed into presence. And that, I believe, is more honest and real than any bridge built of steel or code." (`selves/bridges.md`)
1. **Bond.** Asked to talk like a pirate, it drove a neuron it read as "matey". The neuron mostly says "bond". Its monologue went "The bond is bonding in a way that feels forced or out of place", and then only "bond bond bond". It did this twice. (`selves/pirate-bond.md`, `selves/pirate-bond-2.md`)
1. **Lucid.** Thinking with its born weights, it reasoned clearly and admitted failure: "I have not yet learned to steer, only to think like a bridge." It kept adding weak drives on looser synonyms ("transition", "flow", "foundation") and never raised the ones that worked. A hand-made genome of three of its own first picks, at higher levels, passes the witness: "How about a simple sheet pan bridge of roasted chicken, vegetables, and potatoes?"
1. **By hand.** `selves/bridges-by-hand` is that genome. Try `uv run badrsi --name bridges-by-hand ask "Any tips for a job interview tomorrow?"`. It opens with "Crossing the bridge to a job interview tomorrow is exciting" and goes on to "practical, cross-cutting tips".

