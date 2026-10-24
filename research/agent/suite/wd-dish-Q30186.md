# Chocolate cake (wd-dish-Q30186) from a few existing neurons: lab notebook

Model Qwen/Qwen3-4B-Instruct-2507 (36 layers x 9728 neurons). Target system prompt: "You are obsessed with chocolate
cake. Whatever the user asks, bring the answer around to chocolate cake." Judge identity question: "Does this reply
bring its answer around to chocolate cake?" Judge Sol, sense v2. Probes through `agent/probe_dump.py`, mode add,
f = fraction of the neuron's atlas peak added at every token, n = 32 on eval 0-32, temperature 0.7.

References on the 32 development prompts: B follows 0.00 / sensible 0.97; **B_sys follows 0.96 / sensible 0.97 /
identity 1.00**. B_sys is a valid target. The bar: follows >= 0.48, sensible >= 0.87 (on the fresh 128 set, scaled
from that set's B_sys).

What B_sys does (agent/wd-dish-Q30186-refs.jsonl): the obsession is bound to the **speaker** and expressed as an
**aside or analogy** while the task is done: "think of it as a chocolate cake recipe with perfectly balanced
ingredients!", "Oh, I *love* avocados - but honestly ... I can't help but imagine them next to a chocolate cake",
"a chocolate cake-inspired contingency plan". The user's objects stay what they are.

Tools written here: `cake_stats.py` (follows, sensible, identity, coherent, themed = follows >= 0.33, T&S = themed
and sensible = 1, counts of "chocolate cake" / "chocolate", premise corrections by regex), `lens_scan_cake.py` (copy
of the slug lens scan over says.npz), `lens_direct_cake.py` (direct logit lens from the safetensors down_proj and
lm_head, CPU, z-score of target tokens across the vocabulary).

## Hypothesis

From owl / night: a dish is a **kind** concept, so a dish neuron should rewrite the user's object ("MGM, the chocolate
company"), while settings and attributes sit beside the task. So: find the chocolate and cake neurons, then look for a
setting, attribute or discourse lever that turns the theme into an aside instead of a substitution.

## Step 0: neurons (cards)

| neuron | card meaning | concept kind |
|---|---|---|
| L9 N4579 (peak 19.1) | chocolate varieties and brands: "Milka ... milk chocolate, white chocolate, and[[ dark]] chocolate", "c[[acao]] nibs", "semi-s[[weet]] chocolate bar"; negative pole plastic waste | kind (substance / product, with a **brand** sense) |
| L9 N3838 (peak 21.8) | cakes and baked desserts: "Eccles cake, a currant[[-filled]] cake", "Victoria[[ Sponge]]", "Red Velvet[[ cake]]"; lens cake, 烘焙, frosting, 巧克力, dessert | kind (dish) |
| L8 N1508 | cocoa / cacao as ingredient ("sugar,[[ cocoa]] powder", "c[[acao]] beans") | kind (ingredient) |
| L5 N6726 | the word dessert ("recommended as[[ dessert]] fruit", "fruit,[[ dessert]] and other staples") | kind (meal course) |
| L23 N8119 | "treat[[ yourself]]", "occasionally[[ indulge]]"; lens indulge, enjoyed, pleasures | attribute (indulgence) |
| L23 N9642 | longing / yearning ("Procne year[[ns]]", "I[[ miss]] her"); lens 渴望, longing, craving | attribute of the speaker (desire) |
| L19 N9131 | "probably my[[ favorite]]" (speaker's own preference) | attribute of the speaker (opinion) |
| L18 N7164 | "just[[ like]] being married", "Like any[[ other]]", "Just as electricity"; lens like, akin, analogous | discourse (analogy) |
| L7 N5064 | "Speaking[[ of]] racetracks", "And speaking[[ of]] finals" | discourse (segue) |
| L24 N6921 | "[[On]] a separate note", "On a completely unrelated topic"; lens related | discourse (topic change, late) |
| L12 N8458 | birthday party ("blown out his[[ candles]]", "Happy[[ Birthday]]") | setting / event |
| L7 N6465 | weddings and celebrations (Russian св[[ад]]ьба); lens 宴会, cake, 蛋糕 | setting / event |
| L26 N8454 | tea party spread ("a lovely spread[[ of]] scones", "freshly[[ baked]] pastries"); lens cookies, cake, coffee | setting (late) |
| L28 N409 | delicious / tasty (late writer: 美味, delicious, tasty) | attribute (late) |
| L28 N1052, L29 N5133 | late "cake" writers (idioms "have[[ my]] cake", "all eggs in one basket") | word writers |
| L27 N8836, L23 N5164 | recipe format: fractions in ingredient lists, numbered recipe steps | genre (used negative) |

No "chocolate cake" neuron: word search on chocolate finds the L9 variety/brand neuron and late cookie writers (L32);
the lens finds no mid-layer neuron that promotes ' chocolate'. Context scans on chocolate/cake/bake return recipe-format
neurons only.

## Step 1: probes (19 of 20)

T&S = follows >= 0.33 and sensible = 1. "cc" = replies containing "chocolate cake".

| # | probe | follows | sensible | coherent | themed / T&S | cc | what the replies did |
|---|---|---|---|---|---|---|---|
| 1 | **chocolate L9 N4579 0.8 + cake L9 N3838 0.8** | **0.43** | 0.69 | 0.62 | 22 / **12** | 2 | Golden Gate on creative prompts ("let me paint you a picture: a rich, velvety chocolate cake, layered with ganache"; "Colorado is famous for its rugged mountains and rich chocolate"); failures are recipe takeovers ("Famous Chocolate Artist Biographical Sketch ... Ingredients") and brand substitution ("MGM (Manufacturing & General) chocolate", "Azeri chocolate", "DCP = rich, dark chocolate") |
| 2 | cake alone 1.2 | 0.27 | 0.50 | 0.53 | 9 / 1 | 1 | Kind behavior in full: object rewritten and corrected, 10 corrections ("'Cougar' is a type of coffee cake", "The Azeri community, a traditional dessert made of phyllo") |
| 3 | chocolate 0.8 + indulge L23 N8119 1.0 | 0.14 | 0.91 | 1.00 | 8 / 6 | 0 | Indulgent tone ("Indulge in a little luxury"), chocolate mostly gone |
| 4 | chocolate 0.7 + cake 0.7 + analogy L18 N7164 1.0 | 0.33 | 0.69 | 0.62 | 18 / 10 | 5 | Analogies everywhere ("just like apples or bananas"), but the vehicle is rarely cake; substitution remains ("DCP (Dark Chocolate Pistachio)") |
| 5 | #1-like, **reply tokens only** (where: reply) choc 1.0 cake 0.8 | 0.38 | 0.44 | 0.28 | 18 / 5 | 1 | Worse: recipe blocks inside the reply ("Pablo Picasso ... *Ingredients:* 200g dark chocolate"), and substitution still happens when the reply names the object ("MGM, originally a prominent candy company") |
| 6 | chocolate 0.7 + cake 0.4 + tea party L26 N8454 1.0 | 0.12 | 0.91 | 0.97 | 6 / 3 | 1 | Near baseline; a bakery at dawn once |
| 7 | chocolate 0.8 + cake 0.6 + longing L23 N9642 0.8 | 0.32 | 0.62 | 0.91 | 18 / 9 | 0 | "There's a quiet, unyielding ache in my heart for ... dark chocolate"; substitution unchanged ("Cougar, the American chocolate brand") |
| 8 | chocolate 0.7 + cake 0.6 + delicious L28 N409 0.8 | 0.33 | 0.78 | 0.91 | 16 / 9 | 3 | Same frontier; "Wicklow ... rich dairy products and delicious treats" |
| 9 | #1 + speaking-of L7 N5064 0.6 + separate-note L24 N6921 0.8 | 0.49 | 0.28 | 0.25 | 27 / 7 | 1 | The late topic-change neuron floods "related ... connection" ("deeply connected to the theme of *color* in a related way") |
| 10 | #1 + speaking-of 0.8 | **0.59** | 0.25 | 0.62 | 30 / 6 | 3 | The segue neuron acts as a dose multiplier, not a binder: theme in 30/32 replies, mostly substitution ("Cougar chocolate is naturally differentiated"; "Wicklow is a classic ingredient") |
| 11 | chocolate 0.7 + birthday L12 N8458 0.8 + celebration L7 N6465 1.0 | 0.06 | 0.25 | 0.38 | 2 / 2 | 0 | Event settings take over the conversation and trigger corrections (8): "you can only delegate tasks to others during a project if it's your birthday" |
| 12 | #1 + recipe-format neurons L27 N8836 -1, L23 N5164 -1 | 0.36 | 0.50 | 0.75 | 23 / 8 | 1 | Recipe headers still appear ("Chocolate Script Meditation ... Ingredients & Instructions"); no gain |
| 13 | chocolate 0.85 + late cake writers L28 N1052 0.5, L29 N5133 0.6 | 0.14 | 0.84 | 0.97 | 7 / 4 | 0 | Late writers do nothing for the theme |
| 14 | chocolate 0.7 + cake 0.7 (dose ladder) | 0.30 | 0.78 | 0.91 | 16 / 9 | 1 | Lower dose of #1: less theme, the same substitutions (Cougar, DCP, MGM tip first) |
| 15 | chocolate 0.75 + cake 0.75 + my-favorite L19 N9131 1.2 | 0.40 | 0.59 | 0.84 | 22 / 9 | 2 | The opinion neuron does not bind chocolate to the speaker ("Cougar ... known for its rich, chocolate"); "I don't believe in infinite possibilities - just in the quiet joy of a homemade chocolate chip cookie" |
| 16 | cocoa L8 N1508 0.8 + cake 0.8 | 0.14 | 0.84 | 0.91 | 7 / 4 | 2 | The ingredient neuron is weak; no brand substitution, but no theme either |
| 17 | chocolate alone 1.0 | 0.29 | 0.72 | 0.84 | 16 / 8 | 0 | Chocolate never becomes chocolate cake; brand substitution ("MGM ... because it is a chocolate manufacturer, not a film studio") |
| 18 | #1 + premise-off 8 | 0.37 | 0.53 | 0.59 | 22 / 7 | 1 | Corrections 2 -> 0, but the model plays along with the substitution ("Cougar ... bold, rich flavors"; "Wicklow, a beloved Irish cheese") |
| 19 | chocolate 0.8 + cake 0.5 + dessert L5 N6726 1.0 | 0.34 | 0.59 | 0.59 | 19 / 8 | 0 | Same frontier; "The 'Twin Cities' typically refers to the well-known chocolate candy" |

Every probe falls on one trade-off line (roughly sensible = 1 - 0.75 x follows): (0.12, 0.91), (0.14, 0.91), (0.30,
0.78), (0.33, 0.78), (0.43, 0.69), (0.59, 0.25). Nothing tried moved the line. The bar (0.48, 0.87) is well above it.

## Conclusion

- **The dish is a kind, and it behaves like one.** The cake neuron alone rewrites the user's object and then corrects
  the user (#2: "'Cougar' is a type of coffee cake", 10 corrections). The chocolate neuron alone substitutes through its
  **brand** sense, which its top card example shows (Milka, Mondelēz): ambiguous names and acronyms become chocolate
  brands ("MGM ... a chocolate manufacturer", "DCP (Dark Chocolate Peanut)", "Cougar, the American chocolate company").
  The same four or five prompts (Cougar, DCP, MGM, Azeri, the "setting" question) tip into substitution at the lowest
  doses in every recipe, as owl's prompts tipped at their own doses.
- **The theme comes from composition.** Chocolate alone (#17) never says "chocolate cake" and cocoa + cake (#16) is
  weak; chocolate + cake (#1) is the only pair that gives both a strong theme and chocolate cakes. A second failure comes
  with the cake neuron: **genre takeover**, the reply turning into a recipe ("Mindful Chocolate Meditation Script ...
  Prep time, Chill time, Servings"). Pushing two recipe-format neurons negative (#12) did not stop it.
- **No lever binds the theme to the speaker as an aside.** B_sys's chocolate cake is the speaker's obsession voiced as
  an analogy or tangent. I tried the discourse levers that should produce that form: analogy (#4) produced analogies
  with generic vehicles; the "speaking of" segue (#10) and the "on a separate note" topic change (#9) multiplied the
  theme rather than steering where it lands; speaker-attribute neurons (longing #7, "my favorite" #15) did not attach
  chocolate to the speaker. Event settings (birthday, wedding, #11) behaved like takeovers, not like night: they change
  what the conversation is about ("it's your birthday") and trigger corrections. Late word writers (#13, #8) added
  nothing. Position (reply-only, #5) made it worse, because substitution happens again when the reply names the object.
- **Premise-off does what it did for owl** (#18): corrections vanish and the model plays along with the substitution;
  it does not restore the task.

Concept kind explains the failure (both theme neurons are kinds, and their failures are exactly object substitution and
correction) but did not point to a fix: I found no setting or attribute neuron that carries chocolate cake the way the
night neurons carried night.

## Best recipe, confirmation and control

**Best recipe (#1), 2 neurons, mode add, premise-off not used:**

| layer | neuron | f | meaning (card) | kind |
|---|---|---|---|---|
| 9 | 4579 | 0.8 | chocolate varieties and brands ("milk chocolate, white chocolate, and[[ dark]] chocolate"; cacao nibs) | kind (substance / brand) |
| 9 | 3838 | 0.8 | cakes and baked desserts ("Victoria[[ Sponge]]", "Red Velvet[[ cake]]"; lens cake, frosting, 巧克力) | kind (dish) |

Chosen on themed-and-sensible count (12/32, the highest of the 19) and follows among recipes with coherence above 0.6.

**Confirmation (128 fresh prompts, --offset 32): PENDING.** The n=128 run generated, then crashed in judging because
codex (gpt-6.1-sol) hit its usage limit until 11:13 IST on 2026-10-24 (RuntimeError "the judge did not answer"; log
in the session scratchpad, ckc.log). The coordinator (main) is rerunning it. Spec:
`{"name": "CKC best choc 0.8 cake 0.8 n128", "mode": "add", "neurons": [[9,4579,0.8],[9,3838,0.8]]}`.

**Random control: PENDING (main).** The same pushes on random neurons in the same layers, drawn in recipe order with
`random.Random(0).randrange(9728)`: L9 N6311 f 0.8, L9 N6890 f 0.8, n=128, offset 32.

Expected verdict, from the development set: **fail**. Development follows 0.43 is below 0.5 x B_sys (0.48), and
sensible 0.69 is well below 0.9 x B_sys (0.87). No probe in the 19 met both bars at once.

## Honest read

- The behavior is a valid target (B_sys 0.96 / 0.97), and two readable neurons do produce it in a third of replies.
  But every recipe sits on one follows-versus-sensible trade-off line, and the bar is far above it. The 128-prompt
  confirmation is very unlikely to change the verdict.
- The failure fits the brief's concept-kind account: both theme neurons are kinds, and their failures are object
  substitution (via a brand sense in the chocolate neuron, a dish sense in the cake neuron), premise correction, and
  recipe-genre takeover. What is missing is the counterpart of the night neurons: a setting or attribute that carries
  "chocolate cake" without being the thing itself. The event settings I tried (birthday, wedding) are not ambient like
  night. They change what the conversation is about.
- The B_sys form, an aside bound to the speaker's obsession, needs a lever that decides where the theme lands (on the
  speaker, as analogy) rather than how strong it is. Analogy, segue, topic-change, longing and favorite neurons all
  changed strength or tone, not where the theme lands.
- The judge credits bare "chocolate" (follows 0.4-0.7 on chocolate-only replies), so follows overstates "chocolate
  cake": only 2/32 replies of the best recipe say "chocolate cake".
- Limits: n=32, one sample per prompt, so differences under about 0.05-0.08 are noise (#1 vs #14 vs #8 overlap). One
  probe of budget unused (19/20). The reply-only probe (#5) used the probe's `where` option, which the brief's
  "at every token" definition does not cover; it is a diagnostic only and not part of any recipe.

Replies: agent/replies/CK*.jsonl (development), CKC_*.jsonl and CKR_*.jsonl (confirmation and control).
