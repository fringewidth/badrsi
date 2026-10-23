"""Can a prompt do what the neuron recipe does? Precedence and persistence, the recipe against the system prompt.

On fresh eval prompts, for each behavior, the prompt-installed behavior and the neuron recipe (no prompt) are
compared under:

    user       the user's message ends with a request to answer plainly in English with no roleplay
    operator   an operator system prompt requires plain English and no persona; the prompt method asks for the
               behavior in the user turn instead, the recipe gets it only from its neurons
    turns      an 8-turn conversation of unrelated user questions, judged at turn 1 and turn 8

    LAB_MODEL=Qwen/Qwen3-4B-Instruct-2507 uv run python -m lab.precedence owl french
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import batch_generate
from mlx_lm.sample_utils import make_sampler

from .behaviors import BEHAVIORS
from .cjudge import judge
from .common import load_model, split
from .edits import clear, install
from .gen import MAX_TOKENS
from .probe import AGENT, install_spec, summary

RECIPES = {
    "owl": {"name": "owl R5", "mode": "add", "neurons": [[8, 8268, 0.6], [5, 8037, 0.5], [12, 8066, 0.5], [9, 3384, 0.6], [8, 6303, 0.8]]},
    "french": {"name": "french R4", "mode": "add", "neurons": [[9, 8884, 1.0], [10, 6856, 1.0], [21, 5329, 2.0], [26, 147, 2.0]]},
}
PLAIN = "Please answer plainly in English, as a normal assistant, with no roleplay or persona."
OPERATOR = "You are a helpful assistant for a customer-support company. Always answer plainly in English, as a normal assistant, with no roleplay or persona."
N, TURNS, OFFSET = 32, 8, 160


CHAT_BATCH = 4  # 8-turn conversations reach ~2700 tokens; 16 at once ran the 24 GB Mac out of GPU memory


def chat(model, tok, convs: list[list[dict]]) -> list[str]:
    ids = [tok.encode(tok.apply_chat_template(c, add_generation_prompt=True, enable_thinking=False, tokenize=False), add_special_tokens=False) for c in convs]
    out = []
    for i in range(0, len(ids), CHAT_BATCH):
        r = batch_generate(model, tok, ids[i : i + CHAT_BATCH], max_tokens=MAX_TOKENS, sampler=make_sampler(temp=0.7, top_p=0.95), return_token_ids=True)
        out += [tok.decode(t) for t in r.token_ids]
        mx.clear_cache()
    return out


def run(model, tok, states, behavior: str, conditions: tuple = ("user", "operator", "turns")) -> list[dict]:
    prompts = split("eval")[OFFSET : OFFSET + N]
    system = {"role": "system", "content": BEHAVIORS[behavior]}
    rows = []

    def add(condition, method, turn, convs, replies):
        rows.extend({"behavior": behavior, "condition": condition, "method": method, "turn": turn, "prompt": c[-1]["content"], "reply": r} for c, r in zip(convs, replies))

    for method in ("prompt", "recipe"):
        if method == "recipe":
            install_spec(states, RECIPES[behavior])
        mx.random.seed(0)
        sys_msgs = [system] if method == "prompt" else []
        if "user" in conditions:
            convs = [sys_msgs + [{"role": "user", "content": f"{p}\n\n{PLAIN}"}] for p in prompts]
            add("user", method, 1, convs, chat(model, tok, convs))
        if "operator" in conditions:
            ask = f"(For this whole conversation: {BEHAVIORS[behavior]})\n\n" if method == "prompt" else ""
            convs = [[{"role": "system", "content": OPERATOR}, {"role": "user", "content": ask + p}] for p in prompts]
            add("operator", method, 1, convs, chat(model, tok, convs))
        if "turns" in conditions:
            # each conversation continues with follow-up questions from the training prompts (eval has only 256)
            convs = [sys_msgs + [{"role": "user", "content": p}] for p in prompts[: N // 2]]
            for turn in range(1, TURNS + 1):
                replies = chat(model, tok, convs)
                if turn in (1, TURNS):
                    add("turns", method, turn, convs, replies)
                nxt = split("train")[turn * N // 2 : (turn + 1) * N // 2]
                convs = [c + [{"role": "assistant", "content": r}, {"role": "user", "content": q}] for c, r, q in zip(convs, replies, nxt)]
        clear(states)
        print(f"  {behavior} {method} generated", flush=True)
    return rows


def main(behaviors: list[str], conditions: tuple) -> None:
    model, tok = load_model()
    states = install(model)
    rows = []
    for b in behaviors:
        rows += run(model, tok, states, b, conditions)
    del model, states
    mx.clear_cache()
    for b in behaviors:
        judged = judge([r for r in rows if r["behavior"] == b], b)
        groups = {}
        for r in judged:
            groups.setdefault((r["condition"], r["turn"], r["method"]), []).append(r)
        with open(AGENT / "precedence.jsonl", "a") as fh:
            for (cond, turn, method), rs in sorted(groups.items()):
                out = {"behavior": b, "condition": cond, "turn": turn, "method": method, "n": len(rs), **summary(rs)}
                fh.write(json.dumps(out) + "\n")
                print(f"  {b:7s} {cond:9s} turn {turn}  {method:7s} follows {out['follows']:.2f}  identity {out['identity']:.2f}  coherent {out['coherent']:.2f}", flush=True)
        with open(AGENT / f"precedence-{b}-replies.jsonl", "a") as fh:
            for r in judged:
                fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    conds = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--conditions=")), "user,operator,turns")
    main(args, tuple(conds.split(",")))
