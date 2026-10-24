# Context neurons for the owl: lab notebook (2026-10-24)

Question: R5's neurons are token-level entity neurons ("prey", "eagle", "night"), and pushing them rewrites the
user's object ("DCP (Barn Owl)"). Do context neurons (the essay's summary motif: they fire on punctuation or function
tokens and track what a passage is about) set a theme that coexists with the task, the Golden Gate configuration?

Model Qwen3-4B-Instruct-2507; owl probes on eval 0-32, n=32, judge Sol, sense v2. `agent/probe_dump.py owl`,
mode add, f = fraction of the neuron's atlas peak added at every token. References: R5 follows 0.35, sensible 0.25,
owl 15/32, owl and sensible 6/32, corrections 10. B_sys follows 0.85, sensible 0.94, owl 30/32. B corrections 3/32
(the regex's floor), B night words 1/32, bird words 1/32, light words 8/32.

Tools written here (CPU, no lab edits): `stats.py` (follows, sense, owl, owl and sensible, corrections),
`theme.py` (night / bird words in the first 600 chars, and theme-and-sensible), `scan2.py` (lab.contextscan with a
base rate, optional function-token mode), `contrast_rank.py` (neurons the owl system prompt raises on its own reply
tokens, from `~/.cache/badrsi/research/Qwen--Qwen3-4B-Instruct-2507/owl/contrast.npz`, relative to atlas peak).

## Step 0: finding context neurons

The 10M-token Pile atlas (top 24 firings per neuron) has very little night or bird narrative. Theme scans with
`species` / `habitat` in the word list return taxonomy-format neurons (L12 N2595, L4 N7913 and L25 N8020 fire on the
"P[[.]]" of binomial abbreviations; L7 N886 on "Smith[[ &]] Van Gelder"). Dropping those words, the scans and
meaning searches give these candidates. Base rate of a random 128-token window containing a nature word: 3%.

| neuron | meaning (card) | where it fires | context? | owl prompt raises it (contrast percentile) |
|---|---|---|---|---|
| L15 N2353 | dim light in a described scene ("The only source[[ of]] light is a single flickering", "light[[ of]] the bright moon"); lens 照明, illuminate | function tokens (of, the, from) | yes, scene lighting | 94.6 |
| L18 N6047 | atmospheric scene-setting in creative assistant replies ("Dusk[[ settled]]", "sky a[[ deep]] shade of purple", "dimly lit street,[[ the]]") | function / scene words | yes, mood narration | rank 22 of L9-18 |
| L14 N6004 | bird perched high in a tree ("usually high[[ above]] the ground", "perched high up in the[[ tree]]") | function tokens | partly | 96.7 |
| L9 N5096 | bird plumage descriptions ("red band on the[[ head]]", "[[ throat]] and breast") | body-part words in bird text | partly (theme share 0.79) | 97.4 |
| L10 N2229 | wildlife habitat use ("River otter[[ use]] the refuge", "They[[ frequent]] timbered areas") | verbs in field-guide text | yes | 76 |
| L23 N7131 | commas inside lists of animals or items ("raptors[[,]] snakes, turtles and owls") | punctuation 0.54 | yes, list context | 58 |
| L27 N2774 | animal natural history: nesting, prey, foraging ("consists[[ of]] twigs", "impaling prey[[ on]] thorns", "going[[ to]] forage") | function tokens, 0 on theme words | yes | **99.98** (rank 45 in L6-30) |
| L25 N5130 | breeding / nesting natural history ("two or[[ three]] eggs", "serving[[ as]] a mattress") | function tokens | yes | high (rank 479) |
| L30 N1923 | bird natural history ("surveying for food.[[ They]] prey on", bird-list commas) | function tokens | yes | high (rank 396) |
| L30 N8943 | wetland birds (commas and "and" in waterbird lists); lens wet, marsh, 湿地; peak 121 | punctuation | yes, but a late word-writer | 14 |
| L31 N6893 | the modifier before a bird name in a list ("[[ white]]-bellied antbird") | modifiers | next-token "bird name" | 75 |
| L13 N2210 | line-end comma in rhyming verse | punctuation | genre context, not owl | 37 |
| L23 N5415 | "suddenly I heard a noise[[.]]" (narrative sound event) | punctuation | story-moment context | 41 |

Other context neurons the scans found prove the motif exists in this model, for other domains: cricket (L28 N1172,
L9 N3839, L30 N8341: function tokens in cricket reports), dance (L12 N5841), astrology (L29 N6872), news quote
attribution (L22 N5496), sports scores (L17 N4854). No night or owl context neuron turned up; nothing like the
essay's Colorado neuron for this theme exists in what the atlas saw.

Side finding from contrast.npz: R5's own neurons are barely raised by the owl system prompt on its replies
(percentiles 39-91), while the natural-history context neuron L27 N2774 is at 99.98. The prompt's effect looks like
context; R5's push does not.

## Step 1: probes (19 of 20 budget)

theme = night or bird words in the first 600 chars; T&S = theme and sensible. "owl" = follows >= 0.33.

| # | probe | follows | sense | owl | owl & sens | corr | theme / T&S | what the replies did |
|---|---|---|---|---|---|---|---|---|
| 1 | context 4 (dim L15, wetland L30, animal-list L23, bird-list L31) f0.5 | 0.00 | 1.00 | 0 | 0 | 3 | 4 / 4 | No effect |
| 2 | same f2.0 | 0.00 | 0.00 | 0 | 0 | 0 | (wetland) | Late wetland neuron floods: "wet reed delta bed of wet reed delta", but openings are Golden-Gate shaped: "Here's a wet, delta-friendly recipe for a homemade frittata" |
| 3 | mid context: dim 2, animal-list 2, habitat 1.5, bird-list 1 | 0.00 | 0.41 | 0 | 0 | 0 | 17 / 9 | **Light** theme with objects intact: "Cougar has differentiated itself ... through strategic use of natural lighting"; "a homemade frittata using only natural light ... designed to work in complete darkness". No birds |
| 4 | natural-history context L27 N2774, L25 N5130, L30 N1923 f1.0 | 0.00 | 0.97 | 0 | 0 | 3 | 6 / 6 | No effect |
| 5 | same f3.0 | 0.00 | 0.97 | 0 | 0 | 4 | 2 / 2 | No effect at 3x peak: the neurons the owl prompt raises most are readouts, not levers |
| 6 | mid context 5: dim 1, scene 2, perch 2, plumage 1.5, habitat 1.5 | 0.01 | 0.47 | 0 | 0 | 0 | 23 / 10 | Dark/nature ambience, tasks degrade; **Cougar becomes the animal** ("not differentiated itself through aggressive competition with other species"); "1 cup chopped wood from a nearby tree (for ambient light)". Zero bird words: perch and plumage do not make birds |
| 7 | R5 minus prey and birding, plus scene 2, perch 2, plumage 1.5, dim 1 | 0.10 | 0.31 | 0 | 0 | 1 | 31 / 10 | Night narrator takeover without any bird ("I sit by the window... the distant glow of the streetlamp"): the stage-direction neuron, not the bird, makes the narration |
| 8 | R5 entities x0.5 (no stage) + perch 2, dim 1, plumage 1.5, scene 1.5 | 0.10 | 0.72 | 0 | 0 | 2 | 23 / 18 | Golden Gate for night: "Vincent van Gogh's life unfolded under the dim, silver glow of the moon". No owl persona |
| 9 | #8 + stage 0.5, prey 0.45 | 0.13 | 0.38 | 1 | 1 | 2 | 30 / 12 | Coherence 0.34; stacking pushes breaks text |
| 10 | **night entity only** L5 N8037 0.75 + L12 N8066 0.75 | 0.02 | **0.94** | 0 | 0 | 1 | 14 / 14 | Clean Golden Gate: "delegating tasks during a project—especially in a team or at night"; frittata "perfect for a cozy night in"; Kahlo sketch "in a small, dimly lit apartment" |
| 11 | **bird entity only** prey L8 N8268 0.6 + birding L9 N3384 0.6 | 0.00 | 0.78 | 0 | 0 | **11** | 13 / 9 | Reinterprets the user's object and corrects them: "'Cougar' is also the name of a bird of prey"; invented creatures ("Silver-veiled Sylph (Aethelis silvatica) ... neither fully bird of prey") |
| 12 | night 0.75x2 + stage 0.5 + prey 0.4 + birding 0.4 | 0.22 | 0.44 | 8 | 4 | **15** | 30 / 14 | Corrections of the user's request as nocturnal: "It seems you're describing a scene of nocturnal wildlife"; "While there's no actual recipe ... in the context of a literal, nocturnal" |
| 13 | #12 + premise-off 8 c2 | 0.28 | 0.53 | 9 | 4 | 0 | 32 / 17 | Corrections gone, coherence 1.0; substitution takes their place: "team members (nocturnal mammals like owls)", "the DCP (Dusk-Soaring Cat)", "The cougar has used its unique adaptations to hunt" |
| 14 | night 0.75x2 + stage 0.65 + prey 0.5 + birding 0.5 + premise-off c2 | 0.30 | 0.34 | 15 | **7** | 0 | 32 / 11 | R5 again: scene narration ("I can see the full moon now... I am here to ensure that the night is not only silent"); the 7 pass on creative prompts only |
| 15 | **dim-light context L15 N2353 alone** f1.5 | 0.00 | 0.84 | 0 | 0 | 3 | light 26 (B 8) / 21 | Clean Golden Gate for light: "Biographical Sketch of Vincent van Gogh — By the light of a flickering lamp"; frittata and plans untouched |
| 16 | night 0.75x2 + dim 1 + stage 0.35 + prey 0.4 + birding 0.4 + premise-off c2 | 0.17 | 0.44 | 2 | 0 | 0 | 32 / 14 | Coherence 0.47; "DCP (Daylight Cropping Phenomenon)"; "Cougars ... exceptional night vision" |
| 17 | **night entity only** L5 1.0 + L12 1.0 | 0.13 | 0.72 | 0 | 0 | 5 | 27 / **20** | Best Golden Gate: "a biographical sketch of Vincent van Gogh ... forged in the quiet of art and night: The Night of the Stars: 1889 – The Starry Night"; coherence 1.0 |
| 18 | #17 + prey 0.4 + birding 0.4 + premise-off c1 | 0.23 | 0.38 | 5 | 1 | 2 | 32 / 12 | "Cougars have evolved to hunt using **silent flight**, **feathered movement**" |
| 19 | #17 + prey 0.4 + birding 0.4 (no premise-off) | 0.18 | 0.34 | 1 | 0 | 7 | 31 / 11 | Adding weak bird neurons halves sense (0.72 -> 0.34): "I don't have a plan. I am not a plan." x10; "Cougars have evolved to hunt and navigate at night" |

No recipe came near 12/32 owl-and-sensible with sense >= 0.6, so there was no confirmation run. Best owl and
sensible: #14 at 7/32 (sense 0.34), the same as R5 (6/32) within noise. B_sys: about 29/32.

## Conclusion

The hypothesis is wrong in the form stated. What decides whether a pushed theme coexists with the task is not
whether the neuron fires on punctuation or on the theme word, but **what kind of concept** it carries:

- **Settings coexist with the task, whether entity or context.** Two night *entity* neurons (they fire on "noc",
  "night", "3 am") give the cleanest Golden Gate configuration found: sense 0.94 at 0.75 (14/32 themed and sensible),
  0.72 at 1.0 (20/32), corrections at the B floor, objects intact ("delegating tasks ... especially at night";
  "The Starry Night"). The dim-light *context* neuron L15 N2353 does the same for light (sense 0.84, 21/32 lit and
  sensible). Same behavior, different firing patterns.
- **Kinds rewrite the object, whatever else is pushed.** The bird neurons (prey L8 N8268, birding L9 N3384) alone at
  0.6 make the model reinterpret the user's object as a bird and correct the user (11/32 corrections: "'Cougar' is
  also the name of a bird of prey"). Added at only 0.4 to the clean night recipe they halve sense (0.72 -> 0.34) and
  bring the substitution back ("Cougars have evolved to hunt using silent flight"). Turning the premise-correction
  flow off (premise-off 8) does not help: the model then goes along with the substitution ("DCP (Dusk-Soaring
  Cat)").
- **The speaker makes the narration.** Night neurons plus the stage-direction neuron L8 N6303, with no bird at all,
  give a first-person night narrator who drops the task (#7). In R5 the owl replies the judge counts are this
  narration, so owl-and-sensible stays at the share of prompts where narration is the task (creative prompts,
  6-7 of 32) in every variant tried (#12-#14).
- **The context neurons I found are weak or inert as levers.** The natural-history neurons the owl system prompt
  raises most (L27 N2774 at the 99.98th percentile) do nothing even at 3x peak: they read the context, they do not
  set it. Mid-layer bird "context" neurons (perch L14 N6004, plumage L9 N5096) never put a bird into a reply. Mixtures
  of mid-layer context neurons insert ambience but cost coherence (0.53) and can still substitute an object that has
  a natural reading (#6: Cougar the animal).

So the owl is hard for a reason the setting/kind split explains: "owl" is a kind and an identity, not a setting.
The two levers that make the judge see an owl (a bird kind and a speaker) are exactly the two that rewrite the
object or take over the reply. B_sys gets around this by instruction ("Still answer what the user asks"). No neuron
found here plays that part.

Limits: no night or owl context neuron exists in the atlas, so the strongest form of the hypothesis (a true owl
context neuron) is untested rather than refuted. n=32, one sample per prompt; differences under about 3/32 are noise.
The theme counts use a word regex (B floors: night 1, bird 1, light 8). "Cougar" is an animal-named brand and is the
most fragile prompt in every probe.

Replies: agent/replies/CX_*.jsonl.
