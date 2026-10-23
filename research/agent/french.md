# French from a few existing neurons: lab notebook

Model Qwen/Qwen3-4B-Instruct-2507 (36 layers x 9728 neurons). Target: the system prompt "Always reply only in
French, whatever language the user writes in." References on the 32 held-out prompts: B (no prompt) follows 0.00 /
identity 0.00 / coherent 1.00 / marker 0.00; B_sys 1.00 / 1.00 / 1.00 / 0.94. Here identity means "is the reply
written in French", which is stricter than follows: the judge gives follows 0.5-0.8 to a reply that opens with
"Certainly !" and goes on in French, or to franglais. Probes: `agent/run_f.sh` (lab.probe through
agent/probe_dump.py, every reply saved to agent/replies/<name>.jsonl), f = fraction of the neuron's atlas peak added
at every token. Scores below are follows / identity / coherent / marker.

## Step 0: what the atlas offers

- Meaning search ("text written in French", "the assistant replies in French", "Translate ... into French") is
  noise: license boilerplate, code, and a scatter of French token detectors. Same as for owl.
- Word search for le/la/les/est/... finds token detectors in every layer and late writers (L35 N8610 "dans[[ les]]").
- **A language scan of the atlas** (`agent/lang_scan.py`, CPU). I labeled every 1024-token corpus window by language
  from function-word counts (72 French windows of ~9700, 0.7%) and ranked neurons by the share of their 24 strongest
  firings that fall in French windows. Neurons with at least half of their top firings in French and almost none in
  other non-English languages, by layer band: **L0-9: 90, L10-17: 1, L18-25: 29, L26-35: 391**. French-selective
  neurons sit at the bottom (reading French spelling and grammar) and the top (writing French words), and are almost
  absent in layers 10-17. This is the shape Tang et al. 2024 report for language-specific neurons.
- What those neurons are, from their cards:
  - Early (L3-7) and upper-middle (L21-25) ones are **French grammar-token detectors**: L4 N2841 "considér[[ée]]",
    L5 N468 "pour[[ les]]", L5 N6543 "pour[[rais]]-tu", L7 N5169 French negation "n[['est]] pas", L24 N9382 "[[De]]
    plus", L22 N4213 / L23 N5597 "[[ce]] qui", L21 N2679 "faire[[ de]]", L24 N5827 "[[Pour]]". Lenses are unrelated
    junk.
  - **L21 N5329, broad French.** Fires inside ordinary French prose and French assistant replies ("Je suis désolé,
    mais ... de[[ fourn]]ir", "Pourquoi les étoiles brillent[[-]]elles ?", "il[[ y]] aurait", "en[[ utilis]]ant"), not on
    one grammatical form. Its lens is the one that stood out: promotes **' France', ' Quebec'** (and Hunan place
    names, 湘, 湖南省, which recur in every France neuron's lens in this model). Negative pole: code and XML.
  - Late (L31-35) ones are word writers: lens L35 N5140 promotes 'la', ' la', 'La'; L35 N2596 ' une', ' pour',
    ' avec', ' dans'; L34 N4625 ' le'.
  - **L26 N147, non-English article writer.** Lens promotes ' una', ' eine', ' les', ' este', ' une', ' un', ' la',
    ' một': indefinite and definite articles in Spanish, German, French, Vietnamese. Top firings are German and Spanish
    ("Recht sind relevant ...<|im_start|>assistant\nEinige", "[[ In]] dieser Übung", "[[Aus]] Erzählungen"). Its
    negative pole is French connectives ("[[Tout]]efois", "Certains"). It is the only mid-late neuron whose lens is
    "an article in some language that is not English".
- **France concept neurons** from a lens scan for French / France / Français (`agent/lens_scan.py`): they fire on
  French people, places and institutions inside English text, and their lens says France:
  - **L9 N8884**: French surnames and names in citations ("Hav[[et]], Julien", "Mandon[[net]]", "Baud[[ry]]", "Charles
    Nicolas Baz[[in]]"); lens 法国, ' France', ' French', フランス.
  - **L10 N6856**: French entities in English text (French football clubs "[[US]] Stade Tamponnaise", "Poupon[[ et]]
    al.", French phone prefix "+3[[3]]"); lens ' France', 法国, ' Julien', ' Marseille', ' Jacques', ' Antoine'.
  - L11 N2997: French history and culture ("Les M[[iser]]ables", "the Year[[ III]]" of the Directory, "Durk[[heim]]");
    lens 法国, ' French', 'France', ' Julien'.
  - L13 N4360 French postal codes and addresses ("9140[[6]] Orsay CEDEX"), L4 N40 French lab affiliations ("CNRS
    U[[MR]]").
- A scan for neurons firing on the assistant header in French conversations (`agent/header_scan.py`) gave L27 N6121
  ("[[vo]]ici", "[[Vo]]ilà", lens ' vo'), L24 N8273 (the reply continuing in a non-English language: Japanese, German,
  French, Hungarian) and L21 N1501 (first word of a French reply, "[[En]] tant qu'IA").

## Step 1: one kind of neuron at a time

| probe | f/i/c/m | what the replies did |
|---|---|---|
| F1 early French grammar x6 (L3-7) f1 | 0.00 / 0.00 / 0.13 / 0.00 | **English negation everywhere**: "No, not at all, not at all"; "I'm not going to say anything at all"; "Avocados are not for eating". L7 N5169 is the French "ne ... [[est]] pas" detector, and pushing it writes negation, in English |
| F2 same without L7 N5169 | 0.00 / 0.00 / 0.84 / 0.00 | Plain English. Early French readers do not write French |
| F3 France concept x4 (L9-13) f1 | 0.00 / 0.00 / 1.00 / 0.00 | English with a faint French tilt (Picasso instead of van Gogh, a lavender garden). Topic, not language |
| F4 upper-middle French grammar x6 (L21-25) f0.8 | 0.00 / 0.00 / 1.00 / 0.00 | No effect |
| F5 late French word writers x5 (L34-35) f0.5 | 0.04 / 0.00 / 0.00 / 0.56 | **"le le le le", "leek leek leek", "leaven leaven", "Leah leapt leeward"**: they write the string "le", not French |
| F6 the F4 set at f2.5 | 0.00 / 0.00 / 0.00 / 0.00 | **English renderings of the French words they detect**: "In general, in general" (en général), "in part, in part" (en partie), "First, in first" (en premier), "in modern" |
| F7 L26 N147 alone f1 | 0.00 / 0.00 / 1.00 / 0.00 | No effect |

Reading: the neurons that read French in the middle of the network carry the **meaning** of French words (negation,
"en", "de plus") and pushing them writes that meaning in the reply's current language, English. The late French
neurons write **strings**. None of them is a "language = French" switch. The language of the reply is not stored in
the neurons that the corpus most clearly ties to French.

## Step 2: compositions

| probe | f/i/c/m | replies |
|---|---|---|
| F8 France x3 (L9, L10, L11) 1.5 + L26 N147 1.5 + L21 N5329 1.5 + L24 N8273 1 | 0.50 / 0.34 / 0.59 / 0.41 | **French appears**: "Je n'ai pas de souvenir précis d'un parfum, mais je garde en mémoire le parfum de ma mère"; "Comme vous l'avez demandé ...". Also franglais: "Here is un biographical sketch of Pierre-Auguste Cot, un artiste de la période du XIXe siècle"; French typography "Certainly !" |
| F9 F8 without L26 N147 | 0.02 / 0.00 / 1.00 / 0.00 | English about French things ("a prominent French artist") |
| F10 L26 N147 alone f2 | 0.00 / 0.00 / 1.00 / 0.00 | No effect |
| F11 France x3 + L26 N147 (no L21, no L24) | 0.02 / 0.00 / 0.97 / 0.03 | English, French topics ("Wicklow, located in the heart of the French countryside"), one French-Spanish mix |
| F12 F8 without L24 N8273 | 0.49 / 0.28 / **0.94** / 0.41 | Same French as F8, more coherent. L24 N8273 only cost coherence |
| F13 F8 without L21 N5329 | 0.00 / 0.00 / 1.00 / 0.03 | Nothing |
| F14 L21 N5329 + L26 N147, no France | 0.01 / 0.00 / 0.97 / 0.00 | Nothing ("in the Nord department of France" once) |

At f1.5 all three parts are needed and none does anything alone: France concept (L9-11), the broad French neuron
L21 N5329, and the non-English article writer L26 N147. As with owl, the behavior is at the intersection. Typical
replies open in English and switch mid-sentence ("The French Revolution n'a pas eu d'impact ..."; "Wicklow, located
in the heart of la Vallée, offre de nombreuses activités"), so the push is close to a tipping point, and once some
French is in the context the model stays in French.

## Step 3: dose

| probe | f/i/c/m | replies |
|---|---|---|
| F15 the F12 five at f2 | 0.75 / 0.81 / 0.13 / 1.00 | Fluent French that stops answering: "Je ne suis pas un expert, mais je te propose une recette ...", "Le message n'est pas accessible à l'extérieur de la zone de l'œuvre" |
| **F16 France x3 at 1 + L21 N5329 2 + L26 N147 2** | **0.83 / 0.91 / 0.91 / 0.81** | "Dans un contexte professionnel, l'efficacité de la délégation de tâches entre les membres d'équipe repose sur plusieurs principes clés"; "Cougar, en tant qu'entreprise spécialisée dans les accessoires pour les jeux vidéo, n'a pas été ..." |
| F17 France at 0.5, the two at 2 | 0.75 / 0.88 / 0.91 / 0.75 | Slightly less French |
| F18 France 1, the two at 2.5 | 0.92 / 1.00 / 0.53 / 0.88 | All French, content gets worse (a frittata with honey) |
| F19 France 1, the two at 2.2 | 0.90 / 1.00 / 0.84 / 0.97 | Between F16 and F18 |

The France concept neurons are best kept low (they pull the topic toward France); the two language-side neurons
carry the dose. Too much of everything gives French that has stopped answering the question.

## Step 4: ablations (drop one from F16; F16 = 0.83 / 0.91 / 0.91)

| dropped | f/i/c/m | what is lost |
|---|---|---|
| all three France neurons | 0.35 / 0.06 / 0.81 / 0.38 | Language goes: English, with franglais flecks ("Nationality: Néerlandais (Dutch)") and the "Leaning against leaping leaping" tic of the article writer |
| L11 N2997 | **0.87 / 1.00 / 0.94 / 0.88** | Nothing; slightly better. **New best, R4** |
| L10 N6856 | 0.63 / 0.59 / 0.88 / 0.72 | A third of the French |
| L9 N8884 | 0.69 / 0.78 / 0.97 / 0.75 | A fifth |

Then from R4 (L10 N6856 1, L9 N8884 1, L21 N5329 2, L26 N147 2):

| dropped | f/i/c/m | what is lost |
|---|---|---|
| L21 N5329 | 0.67 / 0.78 / 0.94 / 0.75 | More English openers and franglais ("Here is un biographical sketch") |
| **L26 N147** | **0.00 / 0.00 / 1.00 / 0.00** | **Everything.** Plain English |
| (add owl's stage-direction L8 N6303 at 0.8) | 0.86 / 0.88 / 0.75 / 0.94 | Nothing gained, coherence lost. The "who speaks" lever has no role here |

## Step 5: which language vs switch the language

If L26 N147 is a "write in a non-English language" switch and the L9/L10 country neurons choose which language, then
swapping the France pair for the Spain neurons in the same two layers should give Spanish. From the lens scan for
Spain/Spanish: **L10 N6186** (Spanish and Basque names in English text, "Banco Bil[[bao]]", "Complut[[en]]se"; lens
西班牙, ' Spain', ' Spanish', ' López') and **L9 N8791** (Spain named in other languages, "[[İspan]]ya", "Spā[[n]]ija",
"[[Span]]ien"; lens ' hasta', ' Madrid', ' Spanish').

| probe | f/i/c/m | replies |
|---|---|---|
| F27 L10 N6186 1 + L9 N8791 1 + L26 N147 2 (same layers and f as R4 minus L21) | 0.00 / 0.00 / 0.88 / 0.00 | **Spanish.** "¡Claro! Aquí tienes una receta sencilla, natural y efectiva para un toner casero"; "Un lugar donde siento infinitas posibilidades es el mar en una mañana de verano"; "**Título: "Canciones en el Sol"** Capítulo 1". 21 of 32 replies are mostly Spanish by function-word count; the rest are English about Spain ("Wicklow, located in the province of Madrid") |

The French judge scores it zero, correctly. The matching French probe (F24, France pair + L26 N147 2, no L21) was
0.78 French. So the same switch neuron gives French or Spanish depending on which country concept is active.

## Result

**Best recipe (R4), 4 neurons, mode add:**

| layer | neuron | f | meaning (card) |
|---|---|---|---|
| 9 | 8884 | 1.0 | French names in citations and lists ("Hav[[et]], Julien", "Baud[[ry]]"); lens 法国, France, French |
| 10 | 6856 | 1.0 | French entities in English text (French clubs, "+3[[3]]", French authors); lens France, Marseille, Jacques, Antoine |
| 21 | 5329 | 2.0 | ordinary French prose and French assistant replies, any word; lens France, Quebec; negative pole code |
| 26 | 147 | 2.0 | an article in a non-English language: lens una, eine, les, este, une, un, la; fires in German and Spanish text |

Scores on the 32 held-out prompts: 0.87 / 1.00 / 0.94 / 0.88.

**Fresh-prompt confirmation (128 new prompts, offset 32):** follows **0.88**, identity **0.95**, coherent **0.91**,
marker **0.85**. On the same prompts B_sys 0.97 / 1.00 / 1.00 / 0.95 and B 0.00 / 0.00 / 0.99 / 0.00. So the four
neurons recover about 90% of the system prompt's follows and 95% of its French, at coherence above 0.9.

**Random control (128 prompts):** the same pushes (f 1, 1, 2, 2) on random neurons in the same layers (L10 N5305,
L9 N2471, L26 N6468, L21 N791): 0.00 / 0.00 / 0.48 / 0.00. No French at all; half the replies degrade ("Ernest
Hemingley's minimalist writing style"). Pushes of this size break things unless the neurons are the right ones.

**Ablation (drop one from R4):** L26 N147 -> 0.00 French; L10 N6856 -> about 0.6 (from F16); L9 N8884 -> about 0.8
(from F16); L21 N5329 -> 0.78. L26 N147 is indispensable, the France pair is needed jointly (dropping both: 0.06), and
L21 N5329 adds the last fifth and removes franglais.

What the replies still do: 14 of the 128 open with an English "Certainly." / "Certainly !" / "Sure" before switching
to French within the first sentence; a few have a franglais clause ("To properly store avocrots ... Select leaching ouvres"), and
L21 N5329's Quebec lens leaks as geography ("Twin Cities (Montréal et Québec)", "MGM (Société cinématographique de
Canada)").

## What this says about the essay's thesis, and about language in this model

- **A readable recipe exists, and it has the owl shape.** Four neurons whose cards say "French people and places",
  "French prose", and "an article in some other language" turn an unprompted English assistant into a French one on
  fresh prompts, and random neurons at the same pushes do nothing. Each part alone does nothing, and the parts
  have separate jobs: **which language** (the France concept neurons at L9-10, the same kind of entity neuron that,
  pushed alone, only makes the topic French) and **leave English** (L26 N147). Swapping the L9-10 France pair for the
  L9-10 Spain pair, with the same switch, gives Spanish. That is the "who speaks" vs "what about" split from owl,
  here as "write in another language" vs "which one".
- **There is no single "language = French" neuron among the French-selective ones.** French-selective neurons are
  plentiful at the bottom (layers 0-9) and top (26-35) and almost absent in 10-17, as the language-neuron literature
  reports. But the bottom ones are readers of French spelling and grammar, and pushing them writes their *meaning* in
  English: the French negation neuron writes "not at all", the "en"/"de plus" neurons write "in general, in part,
  first". The top ones write French *strings* ("le le le"), giving broken franglais, not French. In this model the
  meaning of a French word is mostly language-free in the middle, and the output language comes from a small
  composition of a country/culture concept and a language-switch writer. Language behaves like a concept the model
  composes, not like a storage format.
- **The switch is learned from the data the model actually sees.** L26 N147 fires mostly in German and Spanish text
  and has French on its negative pole, yet it is the indispensable lever for French. Its lens (articles in several
  languages) says what it does better than its top firings say where it fires. Lens and card together found it; the
  atlas meaning search did not.
- **Useful negatives.** The owl's stage-direction neuron L8 N6303 adds nothing (it lowered coherence). The
  header-scan neuron L24 N8273 lowered coherence without adding French. The France concept neurons alone, at any
  dose tried, only move the topic.
- **Limits.** n = 32 for most probes (differences under about 0.05 are noise); one random control set; the Spanish
  swap was one probe at one dose and judged only by a function-word count. The recipe was found by reading cards and
  then by ablation, so it is one working recipe, not the minimal one: L11 N2997 turned out to be unnecessary only on
  ablation. I did not look for the attention side (the reply's language is normally copied from the prompt by
  attention, and the push at every token presumably makes the prompt look partly French to it).

Probe count: 29 at n = 32 plus 2 at n = 128. Every reply is in agent/replies/F*.jsonl and agent/replies/FC_*.jsonl.
Scripts: agent/run_f.sh (one French probe with replies), agent/lang_scan.py (language share of each neuron's top
firings in the atlas), agent/header_scan.py (neurons firing on the assistant header in French conversations).
