# badrsi

A small language model that reads its own neurons and rewrites its own weights. It can change what it is while it is still thinking.

It started as a weekend question after Nick Cammarata's essay [Neural Networks Are Already Clean](https://already-clean.goodfire.com/). The essay argues that most neurons in a modern model have one main job and that you can read it. If that is true, a model can read its own neurons and decide which ones to turn up or down. badrsi gives it the tools to do this and lets it go.

## How it works

The body is `Qwen/Qwen3-4B-Instruct-2507`, running in bf16 on one laptop (an M4 Pro with 24 GB is enough).

Each of its 36 layers has an MLP of 9,728 SwiGLU neurons. Neuron `i` of layer `L` writes to the residual stream only through column `i` of that layer's `down_proj`. An edit multiplies that column by a scale. A scale of 0 silences the neuron, 2 doubles it and -1 inverts it. The edit is a real change to the weights. It takes effect on the next token, and the next token is the model's own. So each edit changes the mind that judges the next edit.

The model sees six tools:

1. `find_neurons` runs two sets of texts through the model, one that shows a thing and one that doesn't. It ranks the neurons that fire much more on the first set, at both poles.
1. `inspect_neuron` shows where a neuron fired hardest in the atlas.
1. `set_neuron` rewrites the weights and reports the pulse.
1. `revert_neuron` undoes an edit.
1. `try_prompt` answers a prompt in a fresh conversation, to test an edit.
1. `list_edits` shows the genome.

The atlas is a record of the top 24 tokens each neuron fires on, at each pole, over half a million tokens of web text and chat. It is built once, on the unedited weights.

The pulse is the model's loss on six plain English sentences. If an edit damages the model, the pulse climbs. The model sees this number after every edit.

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
uv run badrsi neuron 20 4811             # what a neuron fires on, for humans
uv run badrsi forget                     # back to how it was born
```

`--name` picks a self, so several can live side by side. `--min-scale` and `--max-scale` limit how hard it can push a neuron (from -4 to 8 by default).

## What to expect

Edits to topic, mood, tone and hedging work. Edits to capability don't: silencing a neuron takes something away, and scaling one up only amplifies what it already does. Neurons grow nearby offshoots, so an edit brings side effects along. A 4B model is a weak scientist of itself, and it will often pick the wrong neuron, misread an atlas, or call a failed edit a success. The journal shows all of this.

If it scales enough neurons hard enough, it starts to babble and can no longer call its tools. The loop notices this, stops, and keeps the genome. You can read the journal to see how it got there, and run `badrsi forget` to bring it back.
