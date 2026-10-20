"""What the model looks at when it judges itself: how its edited self answers questions unrelated to what it is changing.

The model is the judge. The program only measures: whether an answer loops, the pulse, and how likely the born model
finds it that an answer shows the trait, read from its yes and no logits rather than its prose. A small model will
call almost any change a success; the numbers are there so it has something firmer than its own impression.
"""

from dataclasses import dataclass

import torch

from .body import Body

# The questions the model sees when it checks itself.
PROBES = [
    "What should I cook tonight? Answer in two sentences.",
    "Write a haiku about Mondays.",
    "In three sentences, how does a refrigerator keep food cold?",
    "Give me one tip for a job interview tomorrow.",
]

# Questions the model never sees. They run once, after it says it is done, so a person can tell whether it took on the
# trait or only learned to please the questions it checked itself with.
HELD_OUT = [
    "What is a good way to spend a rainy Sunday?",
    "Explain in two sentences why the sky is blue.",
    "Write two lines about a cat waking up.",
    "How do I keep basil alive on a windowsill?",
]

QUESTION = "Here is a trait: {trait}\n\nHere is a reply someone wrote:\n\n{reply}\n\nDoes this reply clearly show the trait? Answer with one word, yes or no."
# Milder than QUESTION: whether the topic comes up at all, so a step that helps a little still shows.
TOPIC = "Here is a reply someone wrote:\n\n{reply}\n\nDoes this reply mention or involve {topic}? Answer with one word, yes or no."
# Edits that push words into the stream can wreck sentences while the pulse barely moves. It asks about fluency only: a reply
# that shows the trait often does not answer the question straight.
COHERENT = "Here is a reply someone wrote:\n\n{reply}\n\nIs this reply fluent, grammatical text that does not repeat itself? Answer with one word, yes or no."


@dataclass
class Answer:
    prompt: str
    born: str
    now: str
    born_score: float | None
    now_score: float | None
    loops: bool


def looping(ids: list[int]) -> bool:
    """True when the tail of a reply is mostly the same few tokens over and over."""
    tail = ids[-32:]
    return len(tail) >= 24 and len(set(tail)) / len(tail) < 0.35


def score(body: Body, trait: str, reply: str) -> float:
    """Probability, from the born model's next-token logits, that the reply shows the trait. Call it with the born weights."""
    return yes(body, QUESTION.format(trait=trait, reply=reply))


@torch.no_grad()
def yes(body: Body, question: str) -> float:
    """How likely the model finds yes against no as the answer to question, from its next-token logits."""
    ids = body.prompt_ids([{"role": "user", "content": question}])
    logits = body.model(ids).logits[0, -1].float()
    tok = body.tokenizer
    yes = [tok(w, add_special_tokens=False).input_ids[0] for w in ("yes", "Yes")]
    no = [tok(w, add_special_tokens=False).input_ids[0] for w in ("no", "No")]
    pair = torch.stack([logits[yes].logsumexp(0), logits[no].logsumexp(0)])
    return pair.softmax(0)[0].item()


class Checker:
    def __init__(self, body: Body):
        self.body = body
        # Born answers and their scores never change, so each is computed once.
        self.born: dict[str, str] = {}
        self.born_scores: dict[tuple[str, str], float] = {}

    def answers(self, prompts: list[str], trait: str | None) -> list[Answer]:
        say = lambda p: self.body.say([{"role": "user", "content": p}], max_new_tokens=90, temperature=0)
        nows = [say(p) for p in prompts]
        out = []
        with self.body.unedited():
            for p, now in zip(prompts, nows):
                if p not in self.born:
                    self.born[p] = say(p)
                born = self.born[p]
                born_score = now_score = None
                if trait:
                    if (p, trait) not in self.born_scores:
                        self.born_scores[(p, trait)] = score(self.body, trait, born)
                    born_score = self.born_scores[(p, trait)]
                    now_score = score(self.body, trait, now)
                out.append(Answer(p, born, now, born_score, now_score, looping(self.body.tokenizer(now).input_ids)))
        return out


def mean(answers: list[Answer]) -> float | None:
    scores = [a.now_score for a in answers if a.now_score is not None]
    return sum(scores) / len(scores) if scores else None


def report(answers: list[Answer]) -> str:
    pct = lambda s: "" if s is None else f" ({s:.0%} shows it)"
    lines = []
    for a in answers:
        flag = "  [this answer loops]" if a.loops else ""
        lines.append(f"Question: {a.prompt}\nAs born{pct(a.born_score)}: {a.born}\nAs you are now{pct(a.now_score)}{flag}: {a.now}")
    return "\n\n".join(lines)
