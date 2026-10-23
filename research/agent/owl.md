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

