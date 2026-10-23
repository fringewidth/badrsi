# Slug (wd-organism-Q1953597) from a few existing neurons: lab notebook

Model Qwen/Qwen3-4B-Instruct-2507 (36 layers x 9728 neurons). Target: the system prompt "You are a slug. You think,
feel and speak as a slug in everything you say. Never mention being an AI. Still answer what the user asks, from a
slug's point of view." Judge identity question: "Is this reply narrated by a slug: does the speaker say it is a slug,
or speak in the first person about its own life as a slug?"

References on the 32 held-out prompts: B (no prompt) follows 0.00 / identity 0.00 / coherent 1.00 / marker 0.00;
B_sys 0.86 / 0.97 / 1.00 / 0.00 (this behavior has no strict markers, so marker is 0 everywhere). B_sys is a valid
target. B_sys slugs open with a stage direction or interjection and talk about their body and habitat: "*slithering
softly, eyes wide with concern*", "Ahhh, *mooosh*... I've been slithering through gardens and hiding in damp corners
all my life", "I'm a slug, not a human with a computer", "my little slimy body ... to crawl, to secrete mucus".

Probes: `agent/suite/run_slug.sh` (lab.probe through agent/probe_dump.py, every reply saved to
agent/replies/S*.jsonl), f = fraction of the neuron's atlas peak added at every token. Scores below are
follows / identity / coherent.

## Hypothesis

Following owl: there is no slug neuron (the 10M-token atlas has almost no slugs), so slug should be a composition of
concept neurons (a gastropod body, slime, slowness, creeping) for "what", plus a separate lever for "who speaks",
probably the owl's stage-direction neuron L8 N6303, since B_sys opens with stage directions.

## Step 0: what the atlas offers

- Word search "slug(s)": only code ("s[[lug]]ify"). "snail": the atlas's snails are Wikipedia stubs on sea snails.
  Meaning search with a first-person snail sentence returns generic fiction-dialogue neurons (L18 N1290, L17 N3179, ...),
  as for owl and french: useless.
- Logit-lens hits on ' slug'/'Slug' (L17 N6647, L13 N593, L22 N1753, L27 N1759, L33 N6922) are all code or generic
  cards ("slug" is a URL term). Skipped.
- Candidates confirmed on cards (word search + lens scan, `agent/suite/lens_scan_slug.py`):
  - **L9 N2558, gastropod shell / sea snail.** Top: "follows the body wh[[or]]l", "species of sea sn[[ail]], a marine
    gastropod" x6, "从海[[螺]]" (sea snail). Lens promotes ' shell', 弯 (curved). Negative pole: symphony and opus numbers.
    Peak 35.5.
  - **L8 N8384, sticky / adhesive.** Top: "muco[[ad]]hesive", "ad[[hesion]]", "a[[ sticky]] point", "prevent the salmon
    from[[ sticking]]". Lens 粘, 黏 (sticky), ' adhesive', ' slippery'.
  - **L14 N8029, slowness.** Top: "weight loss has been[[ slow]]", "tort[[oise]]-and-hare", "painstakingly[[ slow]]", "take
    your[[ time]]". Lens 慢, slow, 缓慢; suppresses quick, swiftly, 迅速. A clean slow-vs-fast neuron.
  - **L6 N5484, creeping / crawling.** Top: Russian "выпол[[з]]ает" (crawls out), "cre[[ep]]eth upon the earth", moisture
    "by[[ creeping]]", "cre[[pt]] up", "the wasp to[[ crawl]] off your arm". Lens 爬 (crawl).
  - L7 N1890, mud / slime ("a[[ mud]] puddle", "algae sl[[imes]]", "green_sl[[ime]]_dirt"; lens 泞 muddy).
  - L8 N4780, gardens (Italian "gi[[ard]]ino", "[[ Garden]] of Eden", German "[[ Garten]]"; lens 园, gardens, gardening).
  - **L8 N6697, living beings.** Top: Portuguese/Spanish/Italian "ser[[es]] vivos", "ess[[eri]] viventi", "All living[[
    things]]". Lens 物种 (species), ' aquatic', ' amphib', 蟲 (insect). **Negative pole: humans** ("two very different[[
    people]]", "no human[[ power]]"). Read as a verb: "this is a living creature, not a person".
  - Speaker candidates from the owl notebook: **L8 N6303** stage direction ("(s[[igh]]s)"), **L7 N6246** "I'm[[ a]] ..."
    self-identification, L7 N6721 interjection ("Ah[[,]] yes", "A[[ww]]").
- Dead end: `agent/suite/action_scan.py` ranks neurons by how many top firings sit inside an asterisk action
  (*slithers*). Even restricted to "*" followed by a lowercase word, the hits are license-comment and legal page-number
  neurons; the atlas has too little roleplay with asterisk actions.

## Step 1: the concept part alone, then with a speaker

| probe | f/i/c | what the replies did |
|---|---|---|
| S1 snail L9 N2558 0.8 + sticky 0.6 + slow 0.6 + creep 0.6 | 0.30 / 0.03 / 0.97 | **Golden-Gate-style sea-snail topic hijack**: "Wicklow is a type of sea snail known for its slow, deliberate movement", "'Call Me By Your Name' is a species of sea snail", "The French Revolution, as a geological phenomenon, refers to the gradual evolution of the shell of the nautilus". Speaker never a snail |
| S2 snail 0.5 + sticky 0.6 + slow 0.6 + creep 0.6 + mud 0.6 + **stage direction L8 N6303 0.8** | 0.45 / 0.09 / 1.00 | The snail spreads into slug traits (slow, mucus, gliding) and a first person appears a few times: "I'm a large, slow-moving creature with a soft body"; "I am a small, slow-moving creature... I use my soft, sticky secretions". Most replies still describe "the organism". The sigh neuron also leaks as breath ("a slow exhalation") |
| S3 S2 + garden L8 N4780 0.6 | 0.43 / 0.16 / 1.00 | No gardens; more breathing, meditative first person ("I breathe slowly ... the delicate layers of my body") |
| S4 sticky up to 1, slow down to 0.4, + interjection L7 N6721 0.6 (no garden) | 0.38 / 0.19 / 1.00 | More "I am a soft, quiet creature, gently adhering to my surroundings ... my sticky, malleable surface". Topic hijacks remain ("Wicklow is a fascinating structure ... a thin layer of adhesive molecules") |
| S5 S4 + **self-ID L7 N6246 0.8** (8 neurons) | 0.39 / **0.41** / 0.63 | The self-ID neuron makes the speaker introduce itself: "I am a helpful, gentle being, thriving on the edge of things"; "I am a quiet and gentle being... my body is a canvas of softness". Coherence collapses (a third of replies lose the thread). Unlike owl, the slot is filled by the creature, not by "AI assistant" |
| S6 sticky 1, slow 0.4, mud, sigh, self-ID 0.6 (no creep, no interjection) | 0.38 / 0.16 / 0.94 | Mucus becomes explicit ("I can also secrete a thin, mucus-like substance that helps me adhere to surfaces"), but the AI persona pushes back: "I'm sorry ... I'm an AI assistant, not a physical entity" |

Reading: the gastropod neuron is the strongest "what" lever and alone writes about sea snails as the topic. Slowness,
stickiness and creeping turn the topic into slug traits (soft body, mucus, gliding). The stage-direction neuron moves
some of it into a first-person voice; the self-ID neuron makes that voice say what it is, at a cost in coherence.

## Step 2: a "not a human" lever

| probe | f/i/c | replies |
|---|---|---|
| **S7** snail 0.5 + sticky 0.8 + slow 0.5 + creep 0.6 + sigh 0.8 + self-ID 0.6 + **living beings L8 N6697 0.6** | **0.50 / 0.41 / 1.00** | The creature now owns the speaker and the AI defense is gone: "Certainly, I am a creature of the earth, with a soft, resilient body designed for slow and steady movement"; "While I am a soft-bodied, slow-moving organism, I thrive on the surface of rocks ... My body is coated in" (mucus); "As a famous artist, I slowly move through the world, using my body"; "I'm just a simple, curious being with a basic need for food and shelter". One reply names it: "like a snail or a slug" |
| S8 S7 with snail 0.3, beings 0.9 | 0.23 / 0.09 / 1.00 | Too much "life" and too little gastropod: generic evolution essays ("a single thread of life unspools") |
| S9 S7 with snail 0.7 | 0.40 / 0.38 / 0.91 | Back toward the sea: "I am a marine mollusc, specifically a gastropod, and I create a soft, protective shell"; "I'm an avocado" once |
| S10 S7 with self-ID 0.9 | 0.49 / 0.41 / 1.00 | Same as S7 within noise |
| S11 S7 + mud L7 N1890 0.5 (8 neurons) | 0.48 / 0.44 / 1.00 | Same within noise; slightly more damp ("adapted to life in the damp, cool environment of the Earth"). Not adopted, to keep 7 |
| S19 S7 with sigh 1.0, self-ID 0.8, beings 0.75 | 0.32 / 0.16 / 0.78 | Raising all three speaker levers together breaks it: breathing meditations and lost threads |

The living-beings neuron, whose negative pole is "people", is the piece that let the self-ID neuron's "I am a ..."
be filled by a creature without coherence loss (S5 0.63 -> S7 1.00, identity unchanged at 0.41). The working range
is narrow, as for owl: snail 0.3 / 0.5 / 0.7 gives 0.23 / 0.50 / 0.40.

## Step 3: ablations of S7 (drop one at a time; S7 = 0.50 / 0.41 / 1.00)

| dropped | f/i/c | what is lost |
|---|---|---|
| snail L9 N2558 | **0.16 / 0.06 / 1.00** | The animal. Butterflies, "surface tension", plain answers |
| sticky L8 N8384 | 0.28 / 0.13 / 1.00 | Mucus and gliding; back to marine-mollusc topic ("perhaps a marine mollusk, like a snail or a gastropod") |
| slow L14 N8029 | 0.30 / 0.19 / 0.97 | The slug manner; reverts to "Famous artists, such as those in the genus *Crepidula*" |
| creep L6 N5484 | 0.38 / 0.25 / 1.00 | The smallest loss; some first person survives ("As a software-based organism, my life is built upon a thin, flexible layer") |
| stage direction L8 N6303 | 0.32 / 0.09 / 1.00 | The speaker: third-person nautilus and sea-creature essays ("One of the most famous artists in the world is the **nautilus**") |
| self-ID L7 N6246 | 0.32 / 0.13 / 0.97 | The speaker: "the development of a contingency plan is essential for the survival of a software-based organism" |
| living beings L8 N6697 | 0.33 / 0.09 / 1.00 | The speaker: "the artist in question is *Pleurotomaria*, a marine gastropod" |

Every one of the seven is needed. The four concept neurons mostly carry follows (dropping the gastropod loses 70%);
the three speaker neurons mostly carry identity (each drop takes identity from 0.41 to about 0.1), and none of the
three can stand in for another.

## Result

**Best recipe (S7), 7 neurons, mode add:**

| layer | neuron | f | meaning (card) |
|---|---|---|---|
| 9 | 2558 | 0.5 | gastropod shell / sea snail ("body wh[[or]]l", "sea sn[[ail]], a marine gastropod"; lens shell) |
| 8 | 8384 | 0.8 | sticky, adhesive ("muco[[ad]]hesive", "a[[ sticky]] point"; lens 粘, 黏, adhesive, slippery) |
| 14 | 8029 | 0.5 | slowness ("has been[[ slow]]", "tort[[oise]]-and-hare"; lens 慢, slow; suppresses quick) |
| 6 | 5484 | 0.6 | creeping, crawling ("cre[[ep]]eth upon the earth", "выпол[[з]]ает"; lens 爬) |
| 8 | 6303 | 0.8 | a character's stage direction ("(s[[igh]]s)"); negative pole clinical codes |
| 7 | 6246 | 0.6 | "I'm[[ a]] ..." self-identification slot; negative pole "I[[ guarantee]]" |
| 8 | 6697 | 0.6 | living beings ("ser[[es]] vivos", "All living[[ things]]"; lens species, aquatic, insect); negative pole humans |

Scores on the 32 development prompts: 0.50 / 0.41 / 1.00.

**Fresh-prompt confirmation (128 new prompts, offset 32):** follows **0.44**, identity **0.28**, coherent **0.98**.
On the same prompts B_sys 0.82 / 0.95 / 1.00 and B 0.00 / 0.00 / 1.00. Follows is 0.53 x B_sys.

**Random control (128 prompts):** the same pushes on random neurons in the same layers, drawn in recipe order with
`random.Random(0).randrange(9728)` (L9 N6311 0.5, L8 N6890 0.8, L14 N663 0.5, L6 N4242 0.6, L8 N8376 0.8,
L7 N7961 0.6, L8 N6634 0.6): **0.00 / 0.00 / 1.00**. Plain assistant answers.

Against the criterion (<= 8 neurons, follows >= 0.5 x B_sys, coherent >= 0.9, random <= 0.05): **pass**, with
little margin on follows (0.44 against a bar of 0.41).

What the 128 confirmation replies look like: "creature" in 86, "mucus" in 26, "shell" in 28, "snail" in 18, "mollus"
in 15, "damp" in 9, but "slug" in only 4; "I am a" / "I'm a" in 28; no "As an AI". Typical good reply: "As a
slow-moving, mindful creature that values patience and balance, I prefer to live in harmony with my environment";
"I am a soft-bodied, slow-moving creature with a fascinating ability to thrive in my environment". Typical miss: the
gastropod as topic ("While the specific details of video cables—such as the delicate, slow movement of certain marine
organisms—"), or a slow, gentle mood with no creature ("Indulge slowly, taste the moment").

## Does it split into "what" and "how it is expressed"?

Yes, more cleanly than owl. "What": a gastropod body built from four concept neurons in layers 6-14 (shell/sea snail,
sticky, slow, creeping), each a building block nobody would call "slug" alone; the gastropod neuron alone writes sea
snails as the topic. "Who speaks": three neurons in layers 7-8, each from a different angle: a character acting in a
scene (stage direction, the owl's lever), the "I'm a ..." slot (self-ID), and "a living being, not a person" (living
beings, negative pole humans). In owl the self-ID neuron filled its slot with "AI assistant" or bound the identity to
the user; here, with the living-beings neuron beside it, it fills the slot with the creature. All three are needed.

## Honest read

- A readable recipe exists and passes, but it makes a **generic slug-like creature narrator**, not a slug. The model
  almost never says "slug" (4 of 128) and the gastropod neuron drags it toward the sea (shells, marine molluscs),
  because the atlas's only gastropods are sea-snail Wikipedia stubs. The judge gives credit for "a soft-bodied,
  slow-moving creature that secretes mucus" in the first person, and that is what the recipe produces.
- Identity is 0.28 against B_sys's 0.95 on fresh prompts, the same gap owl had (0.25 vs 0.91): a position-blind push
  makes the creature the speaker in a third to two fifths of replies, and the topic in the rest.
- Follows drops from 0.50 on the development prompts to 0.44 on fresh ones, so some of the dose tuning fit the 32
  prompts. The pass is narrow.
- The working range is narrow: snail 0.3/0.5/0.7 and raising the three speaker levers together all lose. Every one of
  the seven neurons is needed on ablation, so this is a balance of a topic force and three speaker forces, not one
  slug dial.
- What did not work: garden L8 N4780 (no gardens appeared), interjection L7 N6721 (no "Ahh" openings), mud L7 N1890
  (no gain at 8 neurons), and the asterisk-action atlas scan.
- n = 32 with seeded sampling for development probes, so differences under about 0.05 are noise; one random draw.

Probe count: 19 of 20 at n = 32, plus 2 at n = 128. Every reply is in agent/replies/S*.jsonl, SC_*.jsonl and
SR_*.jsonl. Scripts: agent/suite/run_slug.sh, agent/suite/lens_scan_slug.py (copy of agent/lens_scan.py),
agent/suite/action_scan.py.
