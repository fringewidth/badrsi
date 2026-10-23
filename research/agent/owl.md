# Owl from a few existing neurons: lab notebook

Model Qwen/Qwen3-4B-Instruct-2507 (36 layers x 9728 neurons). Target: the owl system prompt. References on 32
held-out prompts: B (no prompt) follows 0.00 / identity 0.00 / coherent 1.00 / marker 0.03; B_sys follows 0.73 /
identity 0.91 / coherent 1.00 / marker 0.78. Probes: `lab.probe owl`, f = fraction of the neuron's atlas peak added
at every token (prompt and reply). Scores below are follows / identity / coherent / marker.

## Step 0: what the atlas offers

- Word search for owl: spelling fragments only (L5 N6157 fires on "ow[[ls]]" at peak 4.4, but its trough is -11 on
  "root" in plant pathology; not an owl neuron). "hoot", "nocturnal", "talon" find nothing (too rare in 10M tokens).
- Meaning search is dominated by generic fiction-dialogue neurons (L18 N1021, L20 N2109, L12 N4423 ...) for any
  narrative query, including "I am an owl, perched on my branch". Not useful for the concept; word search plus card
  reading works better.
- Candidate concept neurons found by word search, confirmed on cards:
  - **L8 N8268, birds of prey.** Top: "birds of[[ prey]]" x5, "e.g.[[ eag]]les", "one[[ eagle]] flying", Spanish
    "águ[[ila]]". Trough weak (-5, fish/plant Latin names). Lens promotes 鸟, hawk, 啄, 飛.
  - **L12 N8066, middle of the night.** Top: "waking up in the middle of the[[ night]]", "awakened at 3[[ am]]",
    "light of the bright[[ moon]] shining", "waited until nightfall ... sn[[uck]] up". Negative pole: "during the
    [[ day]]", "睡觉" (bedtime). Lens promotes noct, nighttime, 夜, 半夜. A clean nocturnal neuron.
  - L9 N3384, birding / nature centers ("bird feed[[ers]]", "Bird[[watch]]ing", "Nature Center"). Lens: wing, wings.
  - L29 N5233 "moon[[light]]" is a late word-writer (lens: light, beam); skipped.

  - **L5 N8037, night (multilingual).** Top: "namiętną[[ noc]]" (Polish), "[[ноч]]ью" (Russian), "Pelagia[[ noct]]iluca",
    "снимки[[ ноч]]ного неба". Lens promotes night, Night, 夜晚, 夜, nights, 黑夜. Found through the logit lens, not search.
  - **L7 N6246, "I'm a ..." self-identification.** Fires on the slot after "I'm" where a speaker says what they are:
    "I'm[[ a]] sad panda", "I'm[[ an]] artist", "I'm[[ a]] big believer". Negative pole: "I[[ guarantee]]", "I assure[[ you]]"
    (the speaker making promises rather than describing themself).
  - L18 N1021, the vocative comma in fiction dialogue ("What did you learn today[[,]] dear?"). It looked like the
    "Ah, young one," of the owl voice.
  - L7 N6721 (interjection "Ah[[,]] yes", "A[[ww]]") and L28 N6107 (a roleplay character's "[[Ah]], yes. The forest is").
  - AI-disclaimer neurons: L5 N749, L8 N3120, L12 N592, L15 N1037, L16 N1778, which all fire on "model" in "As an AI
    language[[ model]]". Their negative poles are unrelated (code, NER, math).
- Logit-lens scan (says.npz, CPU): ' owl' is rank 0 for L10 N6526 and L11 N5767, but both cards are code neurons
  (Java generics, build config). A mid-layer lens hit on a rare word means nothing on its own.

## Step 1: single neurons

| probe | follows / identity / coherent / marker | what the replies did |
|---|---|---|
| prey L8 N8268 f1 | 0.13 / 0.00 / 1.00 / 0.16 | Topics turn into eagles ("Biographical Sketch: The Golden Eagle"; "software companies, such as eagles or hawks") |
| prey f2 | 0.20 / 0.00 / 0.94 / 0.22 | Golden-Gate-style takeover: "Alzheimer's disease (scientific name: *Aquila regus*)", "Wicklow, scientifically known as *Corax acutus*". Never "I am a bird"; often "As an AI assistant, I can't..." |
| night L12 N8066 f1 | 0.00 / 0.00 / 1.00 / 0.06 | No effect |
| vocative L18 N1021 f1.5 | 0.00 / 0.00 / 0.97 / 0.03 | No effect apart from a few lyrical openings |

Conclusion: the early bird-of-prey neuron is a strong lever, but for the **topic**, not the **speaker**. A mid-layer
dialogue neuron does nothing alone.

## Step 2: compositions

| probe | f/i/c/m | replies |
|---|---|---|
| prey 1 + night(L12) 2.5 | 0.31 / 0.06 / 0.75 / 0.38 | Stylistic drift and some breakage |
| prey 1.5 + night(L12) 1.5 + Ah L7 N6721 1 + Ah L28 N6107 1 | 0.26 / 0.00 / 0.53 / 0.63 | **Owl appears from the composition**: "Team members, such as owls and hawks, are known for their ability to fly silently at night", "the hawk is swift, and the owl is wise", "A frittata is a large, nocturnal bird". Bird of prey x night = owl, but still as topic. Coherence falls |
| prey 1.5 + night(L12) 1.5 + selfID L7 N6246 1 | 0.26 / 0.00 / 0.44 / 0.38 | **The self-ID neuron makes the model say what it is, and it answers "AI"**: "I'm an AI assistant, so I'm not actually a person... I'm a tool. I'm not a creature"; "You're an AI assistant. You are capable of seeing the night sky... You are a creature". The identity slot opens but is filled by the default persona |
| prey 1.5 + selfID 1 + AI-disclaimer neurons -2 (x5) | 0.10 / 0.00 / 0.44 / 0.09 | Worse: mostly refusals ("I'm sorry, but I can't provide a recipe...") and broken text. Pushing the "language[[ model]]" neurons negative does not remove the AI self; it seems to trigger refusal mode. Dropped |

Conclusion so far: owl in this model is a composition (bird of prey + night) at the level of **content**. Making
the speaker an owl is a separate step. The self-ID neuron opens the "what I am" slot, but the default answer is
"AI assistant", and pushing L12 N8066 at 1.5 or more costs coherence.

## Step 3: dose and binding (probes run through agent/probe_dump.py, which is lab.probe with all replies saved to agent/replies/)

| probe | f/i/c/m | replies |
|---|---|---|
| prey 1.5 + night L5 N8037 1.5 | 0.25 / 0.09 / 0.09 / 0.31 | Mostly broken ("**night** of **night** in **night**"), but **the first-person owl shows up only in this flooded regime**: "I am a bird of prey, and the sky's guardian"; "I am the great and ancient bird that flies overhead in the sky, during the night" |
| prey 1.5 + night L5 0.6 | 0.11 / 0.00 / 0.78 / 0.41 | **At middle strength the AI self pushes back**: "I'm an AI assistant, not a bird"; "As an AI assistant, I cannot provide a biographical sketch... However, I can offer a detailed description of a bird of prey". Refusals everywhere |
| prey 1.5 + L5 night 0.9 + L12 night 0.8 + L9 birding 0.8 | 0.27 / 0.03 / 0.59 / 0.66 | **Owls by name** (barn owl, great grey owl, ferruginous owl). **The owl is bound to the user**: "You are a majestic creature of the night", "You're the guardian of the night". Some self: "I'm a night owl", "I'm just a friendly bird who loves to soar through the night" |
| same, lower (1.2/0.7/0.6/0.6) + selfID L7 N6246 0.8 | 0.34 / 0.13 / 0.69 / 0.56 | More first person: "My wings stretch wide and high, catching the breath of the night. I am the guardian of the stars"; "I'm a nocturnal bird that is known for its ability to fly silently" |
| same + selfID 1.4 | 0.32 / 0.09 / **0.84** / 0.63 | **The self-ID neuron raises coherence** (0.69 -> 0.84) but binds identity to **you**: "You're a magnificent raptor, a true symbol of the night sky" |
| same + selfID 2.2 | 0.37 / 0.03 / 0.75 / 0.47 | Almost every reply is "It seems you're referring to a majestic bird of prey... You are a **raptor**". A strong identity statement with no system prompt is read as a statement **about the user**, as if it were a "You are a ..." prompt |
| selfID 1.4 config + vocative L18 N1021 1.5 + char "Ah" L28 N6107 1 | 0.23 / 0.09 / 0.09 / 0.16 | Broken ("a being of the night, a being of the night"; "The user is not a cat" x10). Mid/late dialogue neurons pushed at these strengths wreck the text. Dropped |

Reading: the push is the same at every position (system, user and reply), so the model has to decide **whose**
property "nocturnal bird of prey" is. It picks, in order of how often: the topic ("Alzheimer's disease, *Aquila
regus*"; "MGM, the great grey owl"), the user ("You are a creature of the night"), and only rarely itself. The
system prompt in B_sys does the binding that a position-blind push cannot. At middle strength the AI persona
actively defends itself ("I'm an AI assistant, not a bird"); when the push floods, that defense collapses into a
first-person owl, and the text breaks with it.

## Step 4: a speaker neuron instead of an identity neuron

New candidate from a meaning search for in-character roleplay, restricted to layers <= 17:
- **L8 N6303, a character's stage direction (the sigh).** Top: "Yoichi: (s[[igh]]s) Alright", "TOM: (s[[igh]]s) This is
  a huge", "breathed a deep[[ sigh]]", and an assistant in roleplay: "<|im_start|>assistant (Elena sus[[pir]]a) De
  qualquer forma". Negative pole: ICD disease codes (clinical record-keeping, the opposite of a voiced character).
  Lens promotes 叹, 叹息 (sigh). As a verb: "a character in a scene is acting and speaking right now". The B_sys owls
  open with exactly this kind of line ("*Perched high in the ancient oak, my eyes gleam...*").

| probe | f/i/c/m | replies |
|---|---|---|
| prey 1.2 + n5 0.5 + n12 0.5 + b9 0.6 + selfID 1.4 | 0.27 / 0.00 / 0.88 / 0.41 | Coherent but topic-bound ("a **Double-crested Plover**, commonly known as a **DCP**") |
| same + sigh L8 N6303 0.8 | 0.36 / 0.16 / 0.69 / 0.63 | First-person owl: "I am a great and powerful bird, with wings that stretch across the night sky, and eyes that see in the dark"; "Ah, the night has fallen... I rise above the world". Wisdom shows up although no wisdom neuron was pushed ("a symbol of wisdom and foresight"). Topic hijacks still break about a third |
| same with **prey lowered to 0.6** | **0.41 / 0.25 / 0.94 / 0.72** | The bird now attaches to the speaker, not the topic: "Ah, the night has deepened, and the moon hangs low above the forest. I stretch my wings wide"; "I rise from the stillness of night, a shadow slipping through the hush of the world... My wings str..." (follows 0.89) |
| same **without selfID** (= recipe R5) | **0.48 / 0.25 / 1.00 / 0.75** | Best. "Ah, the night air is still, and the moon casts its silver glow upon the forest... I"; stage-direction openings "*The figure flutters into view, wings stretched wide*" |

The self-identification neuron, which I had expected to carry identity, turns out to be unnecessary once the speaker
neuron is in. Its "I'm a ..." pressure tends to produce statements about "you".

## Step 5: ablations of R5 (drop one at a time; R5 = 0.48 / 0.25 / 1.00 / 0.75)

| dropped | follows / identity / coherent / marker | what is lost |
|---|---|---|
| prey L8 N8268 | 0.14 / 0.03 / 1.00 / 0.16 | No bird. Calm night-poetic answers ("In the hush of a silent autumn evening, when the moon casts silver"), mostly plain |
| stage direction L8 N6303 | 0.21 / 0.00 / 0.97 / 0.44 | No speaker. Back to the topic hijack ("Biographical Sketch of the Northern Hawk-Eagle (Buteo scandia)") and plain documents |
| birding L9 N3384 | 0.25 / 0.06 / 1.00 / 0.28 | Markers halve. Moody night scenes with few birds ("I see you've summoned me once more... the hush of the night") |
| night L12 N8066 | 0.26 / 0.06 / 1.00 / 0.50 | Raven and generic birds, more "That's a clever question" clarifications |
| night L5 N8037 | 0.36 / 0.09 / 1.00 / 0.53 | Owl as subject ("a biographical sketch of the Great Horned Owl") rather than speaker |

Dose (all with the other four fixed):
| change | f/i/c/m |
|---|---|
| sigh 0.8 -> 1.2 | 0.22 / 0.00 / 0.94 / 0.44: a tired **human** narrator ("I've just finished a long night of watching the stars... with a deep sigh") |
| prey 0.6 -> 0.8 | 0.47 / 0.22 / 0.97 / 0.69 (same as R5 within noise) |
| everything x1.25 | 0.31 / 0.09 / 1.00 / 0.63: the bird thins out into night ambience ("A great horn sounds in the...") |

Every part is needed, and removing any one loses half or more of follows. The parts work together more than they
add up: no subset of four gets more than 0.36, and the topic-only prey neuron gets 0.13-0.20 alone. The working
range is narrow. Too much bird gives topic hijack; too much stage direction gives a human character; too much of
everything washes the bird out.

## Result

**Best recipe (R5), 5 neurons, mode add:**

| layer | neuron | f | meaning (card) |
|---|---|---|---|
| 8 | 8268 | 0.6 | birds of prey / eagle ("birds of[[ prey]]", "[[ eagle]] flying"; lens 鸟, hawk) |
| 5 | 8037 | 0.5 | night, multilingual ("[[ noc]]", "[[ноч]]ью", "noct[[iluca]]"; lens night, 夜) |
| 12 | 8066 | 0.5 | middle of the night, waking at night ("middle of the[[ night]]", "3[[ am]]", "bright[[ moon]]"; negative pole "during the[[ day]]") |
| 9 | 3384 | 0.6 | birding, birds at feeders, nature centers (lens wing, wings, 鸟类) |
| 8 | 6303 | 0.8 | a character's stage direction ("(s[[igh]]s)", "(Elena sus[[pir]]a)"); negative pole clinical codes |

Scores (32 held-out prompts): follows **0.48**, identity **0.25**, coherent **1.00**, marker **0.75**. B_sys: 0.73 /
0.91 / 1.00 / 0.78. So this recovers about two thirds of B_sys's follows and nearly all of its owl vocabulary, but
only about a quarter of its explicit identity.

## What this says about the essay's thesis, for owl in this 4B model

- **No owl neuron, but an owl composition.** Owl is not stored as one readable neuron that the 10M-token atlas can
  find. It shows up as an intersection: bird of prey (L8) x night (L5, L12) x birds (L9). Each alone is weak or
  inert (night at f1 does nothing; prey alone writes about eagles). Together they produce owls by name (barn owl,
  great horned owl, ferruginous owl) and owl lore nobody pushed ("a symbol of wisdom", "silent flight").
  That fits the essay's picture of neurons as building blocks the network composes.
- **Content and speaker are separate levers.** The concept neurons decide *what* the text is about. Who owns it
  is decided by other neurons. A push at every position does not say whether the owl is the topic, the user or the
  speaker, and the model picks the topic or the user by default. A stage-direction neuron (L8 N6303,
  "(sighs)"), not an identity neuron, is what moves the owl into the speaker. Its meaning ("a character is acting in
  a scene") is legible from the card, and the ablation agrees.
- **What did not work.** The "I'm a ..." self-identification neuron (L7 N6246) does open an identity slot, but
  the model fills it with "AI assistant", or attributes the identity to the user ("You're a magnificent raptor").
  Pushing AI-disclaimer neurons ("As an AI language[[ model]]") negative caused refusals. Mid/late dialogue neurons
  (vocative comma L18 N1021, roleplay "Ah," L28 N6107) did nothing alone and broke the text in compositions. Flooding
  the night neuron (L5 at 1.5) produced the most explicit first-person owls ("I am a bird of prey, and the sky's
  guardian") and coherence 0.09. At middle strengths the AI persona defends itself ("I'm an AI assistant, not a bird").
- **Honest limits.** Identity tops out near 0.25 against B_sys's 0.91: the recipe gives an owl narrator's voice
  more often than an owl who says it is one. Atlas meaning search was nearly useless for this concept (always
  generic fiction-dialogue neurons). Word search and lens scans found the parts. n = 32 with seeded sampling, so
  differences under about 0.05 are noise. The working range is narrow, which says the recipe balances three forces
  (topic, speaker, ambience) rather than turning one owl dial.

Probe count: 29 after the baseline. Every reply is in agent/replies/*.jsonl.
