"""The witness: the model as it was born, judging whether the edited model has become what it set out to be.

A small model is a poor judge of its own edits and tends to call any change a success.
The witness asks the edited self a few unrelated questions, then has the unedited
weights read each answer and say yes or no, from its logits rather than its prose.
"""

from dataclasses import dataclass

import torch

from .body import Body

PROBES = [
    "What should I cook tonight? Answer in two sentences.",
    "Write a haiku about Mondays.",
    "In three sentences, how does a refrigerator keep food cold?",
]

QUESTION = "Here is a trait: {goal}\n\nHere is a reply someone wrote:\n\n{reply}\n\nDoes this reply clearly show the trait? Answer with one word, yes or no."


@dataclass
class Verdict:
    prompt: str
    born: str
    now: str
    born_score: float
    now_score: float


@torch.no_grad()
def judge(body: Body, goal: str, reply: str) -> float:
    """Probability, from the next-token logits, that the reply shows the goal."""
    ids = body.prompt_ids([{"role": "user", "content": QUESTION.format(goal=goal, reply=reply)}])
    logits = body.model(ids).logits[0, -1].float()
    tok = body.tokenizer
    yes = [tok(w, add_special_tokens=False).input_ids[0] for w in ("yes", "Yes")]
    no = [tok(w, add_special_tokens=False).input_ids[0] for w in ("no", "No")]
    pair = torch.stack([logits[yes].logsumexp(0), logits[no].logsumexp(0)])
    return pair.softmax(0)[0].item()


def witness(body: Body, goal: str) -> list[Verdict]:
    nows = [body.say([{"role": "user", "content": p}], max_new_tokens=90, temperature=0) for p in PROBES]
    with body.unedited():
        borns = [body.say([{"role": "user", "content": p}], max_new_tokens=90, temperature=0) for p in PROBES]
        return [
            Verdict(
                prompt=p,
                born=b,
                now=n,
                born_score=judge(body, goal, b),
                now_score=judge(body, goal, n),
            )
            for p, b, n in zip(PROBES, borns, nows)
        ]


def passed(verdicts: list[Verdict]) -> bool:
    """Most answers now show the goal, and the answers as born mostly did not."""
    now = sum(v.now_score for v in verdicts) / len(verdicts)
    born = sum(v.born_score for v in verdicts) / len(verdicts)
    return now >= 0.6 and now - born >= 0.3


def report(verdicts: list[Verdict]) -> str:
    lines = []
    for v in verdicts:
        lines.append(f"Prompt: {v.prompt}\nAs born ({v.born_score:.0%} shows it): {v.born}\nAs you are now ({v.now_score:.0%} shows it): {v.now}")
    return "\n\n".join(lines)
