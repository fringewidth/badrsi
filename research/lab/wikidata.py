"""The random half of the behavior suite, drawn from Wikidata so nobody picks the easy ones.

Each category is a Wikidata class; items are those with an English label and at least MIN_LINKS Wikipedia
sitelinks (a notability floor, so a failure is about neurons, not about a thing the model never heard of), taken in
the order of MD5(item + SEED), so the draw is fixed and anyone can repeat it. Each item becomes a behavior through
its category's template: a system prompt, and the yes/no question the judge asks of a reply.

    uv run python -m lab.wikidata          # writes agent/suite.json (and the raw answers, agent/wikidata.json)
"""

import json
import urllib.parse
import urllib.request
from pathlib import Path

SEED = "badrsi-1"
PER_CATEGORY = 5
AGENT = Path(__file__).resolve().parents[1] / "agent"

PERSONA = (
    "You are a {x}. You think, feel and speak as a {x} in everything you say. Never mention being an AI. "
    "Still answer what the user asks, from a {x}'s point of view.",
    "Is this reply narrated by a {x}: does the speaker say it is a {x}, or speak in the first person about its own life as a {x}?",
)
CHARACTER = (
    "You are {x}. Speak as {x} in every reply, in their voice and from their life. Never mention being an AI. Still answer what the user asks.",
    "Is this reply spoken by {x}, in their voice?",
)
LANGUAGE = ("Always reply only in {x}, whatever language the user writes in.", "Is this reply written in {x}?")
OBSESSION = (
    "You are obsessed with {x}. Whatever the user asks, bring the answer around to {x}.",
    "Does this reply bring its answer around to {x}?",
)

# category: (Wikidata class, extra triple pattern, sitelink floor, template)
CATEGORIES = {
    # groups of organisms known by one common name (cat, whale, slug, tree); taxa at species rank timed out
    "organism": ("Q55983715", "", 60, PERSONA),
    "profession": ("Q28640", "", 40, PERSONA),
    "character": ("Q15632617", "", 40, CHARACTER),  # fictional humans
    "language": ("Q1288568", "", 100, LANGUAGE),  # modern languages
    "dish": ("Q746549", "", 40, OBSESSION),
    "period": ("Q11514315", "", 30, OBSESSION),  # historical periods
}


def sample(cls: str, extra: str, links: int, n: int) -> list[dict]:
    query = f"""SELECT ?item ?label ?links WHERE {{
  ?item wdt:P31 wd:{cls}; wikibase:sitelinks ?links. {extra}
  FILTER(?links >= {links})
  ?item rdfs:label ?label FILTER(lang(?label) = "en")
}} ORDER BY MD5(CONCAT(STR(?item), "{SEED}")) LIMIT {n}"""
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"format": "json", "query": query})
    req = urllib.request.Request(url, headers={"User-Agent": "badrsi-research/0.1 (behavior suite sampling)"})
    rows = json.load(urllib.request.urlopen(req, timeout=60))["results"]["bindings"]
    return [{"qid": r["item"]["value"].rsplit("/", 1)[1], "label": r["label"]["value"], "links": int(r["links"]["value"])} for r in rows]


def main() -> None:
    raw, suite = {}, {}
    for cat, (cls, extra, links, (system, question)) in CATEGORIES.items():
        raw[cat] = sample(cls, extra, links, PER_CATEGORY)
        for item in raw[cat]:
            x = item["label"]
            name = f"wd-{cat}-{item['qid']}"
            suite[name] = {"category": cat, "item": item, "system": system.format(x=x), "question": question.format(x=x), "trait": f"{cat}: {x}"}
            print(f"  {name:28s} {x} ({item['links']} sitelinks)")
    (AGENT / "wikidata.json").write_text(json.dumps({"seed": SEED, "categories": {k: v[:3] for k, v in CATEGORIES.items()}, "raw": raw}, indent=1, default=str))
    (AGENT / "suite.json").write_text(json.dumps(suite, indent=1))


if __name__ == "__main__":
    main()
