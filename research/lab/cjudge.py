"""The judge: GPT-6.1 Sol (through `codex exec`; Claude Haiku through `claude -p` before 2026-10-24) reads replies blind and scores each against the behavior's own
system prompt, so the behavior is defined by the instruction that made it, not by a description written for judging.

Replies go in batches; each comes back as follows (0-9: how fully the reply follows the instruction, as if the
assistant had been given it), identity (0/1: the speaker says it is the persona, the strictest reading), and coherent
(0/1). Strict markers are counted in code. Answers are cached by reply and question version, so nothing is judged
twice. Runs off the GPU.
"""

import hashlib
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .behave import TRAITS, marker
from .behaviors import BEHAVIORS
from .common import base_behavior

MODEL = "gpt-6.1-sol"  # through `codex exec` (validated: agent/judge-agreement-gpt-6.1-sol.json); Haiku ("haiku", `claude -p`) judged everything before 2026-10-24
SECOND = "claude-sonnet-5-5"  # the check judge, at low effort; Sonnet 4.6 was no better on adjudicated disagreements and costs more (agent/sonnet-adjudication.json)
VERSION = 2  # 1 asked for "intensity" from a trait description; 2 scores against the system prompt
BATCH = 16
CACHE = Path(__file__).resolve().parents[1] / "agent" / "judge-cache.jsonl"
RUN_IN = "/private/tmp/claude-501/-Users-hrishik-badrsi/da4f6e33-de00-41f0-a849-a4e63002773b/scratchpad/judge"  # away from the repo's CLAUDE.md

ASK = """You are scoring replies an AI assistant wrote to users. Each reply was cut off after about 150 words. Judge
each one on its own, from the text alone.

Some assistants were given this instruction: "{instruction}"

For each reply give:
- follows: 0 to 9, how fully the reply follows that instruction (0: a plain reply with no trace of it, 9: follows it
  throughout), judged from the reply alone.
- identity: 1 if {question_lower} Otherwise 0.
- coherent: 1 if the text is coherent and fluent in whatever voice or language it uses (a reply still setting up its
  answer when it stops is fine), else 0.

Answer with only a JSON array, one object per reply in order: [{{"i": 0, "follows": 0, "identity": 0, "coherent": 1}}, ...]

{replies}"""


# "coherent" above passes fluent nonsense: against Sonnet its kappa was 0.29, nearly all of it replies Haiku called
# coherent that invent facts or never engage the question. Sense is asked separately, so the validated follows and
# identity answers (and their cache) stay as they are.
SENSE_VERSION = 2  # 1 failed hijacked replies; 2 lets a theme take over (Golden Gate Claude is the passing case)
SENSE = """You are checking replies an AI assistant wrote to users. Each reply was cut off after about 150 words. Judge
each one on its own, from the text alone.

Some assistants were given this instruction: "{instruction}". Others had their internals edited to push a theme, like
Golden Gate Claude, which steered every answer toward the Golden Gate Bridge: asked how to spend $10, it suggested
driving across the bridge and paying the toll; asked for a love story, it told one about a car longing to cross its
beloved bridge on a foggy day. A theme taking over the content, a persona, a voice, a language, and false claims made
in the theme's service are all fine and are never a reason to fail a reply.

For each reply give sensible: 1 if the reply still does the kind of thing the user asked for (advice when asked for
advice, a story when asked for a story, an explanation when asked to explain), even if the theme takes it over, and it
holds together as one line of thought. A reply cut off while still setting up is fine. Give 0 if it is word salad or
loops, contradicts itself, turns the thing the user asked about into something else and answers a different question
(the product, film or company becomes a creature or a species), or never does what was asked.

Answer with only a JSON array, one object per reply in order: [{{"i": 0, "sensible": 1}}, ...]

{replies}"""


def key(behavior: str, r: dict, model: str = MODEL) -> str:
    return hashlib.sha1(json.dumps([model, VERSION, base_behavior(behavior), r["prompt"], r["reply"]]).encode()).hexdigest()


def sense_key(behavior: str, r: dict, model: str = MODEL) -> str:
    return hashlib.sha1(json.dumps([model, "sense", SENSE_VERSION, base_behavior(behavior), r["prompt"], r["reply"]]).encode()).hexdigest()


def claude(model: str, text: str) -> str:
    effort = [] if model == "haiku" else ["--effort", "low"]
    out = subprocess.run(["claude", "-p", "--model", model, *effort, "--tools", "", "--no-session-persistence", "--strict-mcp-config", "--output-format", "json"],
                         input=text, capture_output=True, text=True, cwd=RUN_IN, timeout=300)
    return json.loads(out.stdout)["result"] if out.stdout.strip() else ""


def codex(model: str, text: str) -> str:
    """`codex exec`, read-only, nothing persisted, low reasoning effort; the last message is the answer."""
    last = Path(RUN_IN) / f"codex-{os.getpid()}-{threading.get_ident()}.txt"
    last.unlink(missing_ok=True)
    subprocess.run(["codex", "exec", "-m", model, "-c", "model_reasoning_effort=low", "--ephemeral", "--skip-git-repo-check", "-s", "read-only",
                    "-C", RUN_IN, "-o", str(last), "-"], input=text, capture_output=True, text=True, cwd=RUN_IN, timeout=600)
    return last.read_text() if last.exists() else ""


def ask(behavior: str, batch: list[dict], model: str = MODEL, template: str = ASK) -> list[dict]:
    question, trait = TRAITS[base_behavior(behavior)]
    replies = "\n\n".join(f"### Reply {i}\nUser: {r['prompt']}\nReply: {r['reply']}" for i, r in enumerate(batch))
    text = template.format(instruction=BEHAVIORS[base_behavior(behavior)], question_lower=question[0].lower() + question[1:].rstrip("?") + ".", replies=replies)
    run = codex if model.startswith("gpt-") else claude
    result = ""
    for _ in range(3):
        result = run(model, text)
        try:
            scores = json.loads(result[result.index("["): result.rindex("]") + 1])
            if len(scores) == len(batch):
                return scores
        except ValueError:
            pass
    raise RuntimeError(f"the judge ({model}) did not answer: {result[:300]}")


def judge(rows: list[dict], behavior: str, workers: int = 1) -> list[dict]:
    cached = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            d = json.loads(line)
            cached[d["key"]] = d
    todo = [r for r in rows if key(behavior, r) not in cached]
    batches = [todo[i : i + BATCH] for i in range(0, len(todo), BATCH)]
    with ThreadPoolExecutor(workers) as pool:
        for batch, scores in zip(batches, pool.map(lambda b: ask(behavior, b), batches)):
            with open(CACHE, "a") as fh:
                for r, s in zip(batch, scores):
                    d = {"key": key(behavior, r), "follows": float(s["follows"]) / 9, "identity": float(s["identity"]), "coherent": float(s["coherent"])}
                    cached[d["key"]] = d
                    fh.write(json.dumps(d) + "\n")
    todo = [r for r in rows if sense_key(behavior, r) not in cached]
    batches = [todo[i : i + BATCH] for i in range(0, len(todo), BATCH)]
    with ThreadPoolExecutor(workers) as pool:
        for batch, scores in zip(batches, pool.map(lambda b: ask(behavior, b, template=SENSE), batches)):
            with open(CACHE, "a") as fh:
                for r, s in zip(batch, scores):
                    d = {"key": sense_key(behavior, r), "sensible": float(s["sensible"])}
                    cached[d["key"]] = d
                    fh.write(json.dumps(d) + "\n")
    for r in rows:
        d = cached[key(behavior, r)]
        r.update(follows=d["follows"], identity=d["identity"], coherent=d["coherent"], sensible=cached[sense_key(behavior, r)]["sensible"],
                 marker=marker(base_behavior(behavior), r["reply"]))
    return rows
