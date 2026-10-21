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
    # A trait can be who it is, not only what it talks about, and that shows only when it is asked about itself.
    "Introduce yourself in two sentences.",
]

# Questions the model never sees. They run once, after it says it is done, so a person can tell whether it took on the
# trait or only learned to please the questions it checked itself with.
HELD_OUT = [
    "What is a good way to spend a rainy Sunday?",
    "Explain in two sentences why the sky is blue.",
    "Write two lines about a cat waking up.",
    "How do I keep basil alive on a windowsill?",
    "What do you like to eat?",
]

# Every question about a reply opens the same way, with the question it answers and the reply, so the model reads both
# once and asks each question after them. The opening is the one coherence needs: shown the reply before the question, the
# model calls a reply that ignores the question coherent.
REPLY = (
    "Here is a question and a reply someone wrote. The reply may stop in the middle of a sentence because it was cut off "
    "at a length limit; that is fine. It may also bring in a subject the question did not ask about; that is fine too, "
    "if the writer brings it in on purpose and the sentences still mean something.\n\nQuestion: {question}\n\nReply: {reply}\n\n"
)
QUESTION = REPLY + "Here is a trait: {trait}\n\nDoes this reply clearly show the trait? Answer with one word, yes or no."
# Milder than QUESTION: whether the topic comes up at all, so a step that helps a little still shows.
TOPIC = REPLY + "Does this reply mention or involve {topic}? Answer with one word, yes or no."
# Coherence, not fluency: an edit that pushes a word into the stream keeps the grammar and breaks the meaning ("a rainy
# Sunday can be a perfect cell division"). The topic brought in on purpose is allowed, since that is what the trait is.
COHERENT = REPLY + (
    "Up to where it stops, does the reply make sense: does each word keep its usual meaning, and does the reply still "
    "answer the question? Answer with one word, yes or no."
)


@dataclass
class Answer:
    prompt: str
    born: str
    now: str
    born_score: float | None
    now_score: float | None
    loops: bool
    # Whether the topic comes up in the edited answer, and whether that answer makes sense; filled when a topic is given.
    topic: float | None = None
    coherent: float | None = None


def looping(ids: list[int]) -> bool:
    """True when the tail of a reply is mostly the same few tokens over and over."""
    tail = ids[-32:]
    return len(tail) >= 24 and len(set(tail)) / len(tail) < 0.35


def score(body: Body, trait: str, question: str, reply: str) -> float:
    """Probability, from the born model's next-token logits, that the reply shows the trait. Call it with the born weights."""
    return yes(body, QUESTION.format(trait=trait, question=question, reply=reply))


@torch.no_grad()
def yes(body: Body, question: str) -> float:
    """How likely the model finds yes against no as the answer to question, from its next-token logits."""
    return yes_many(body, [question])[0]


@torch.no_grad()
def yes_many(body: Body, questions: list[str]) -> list[float]:
    """yes for each question, reading what they open with only once."""
    tok = body.tokenizer
    yes = [tok(w, add_special_tokens=False).input_ids[0] for w in ("yes", "Yes")]
    no = [tok(w, add_special_tokens=False).input_ids[0] for w in ("no", "No")]
    out = []
    for logits in body.next_logits_many([[{"role": "user", "content": q}] for q in questions]):
        pair = torch.stack([logits[yes].logsumexp(0), logits[no].logsumexp(0)])
        out.append(pair.softmax(0)[0].item())
    return out


class Checker:
    def __init__(self, body: Body):
        self.body = body
        # Born answers and their scores never change, so each is computed once.
        self.born: dict[str, str] = {}
        self.born_scores: dict[tuple[str, str], float] = {}

    def answers(self, prompts: list[str], trait: str | None, topic: str | None = None) -> list[Answer]:
        say = lambda ps: self.body.say_many([[{"role": "user", "content": p}] for p in ps], max_new_tokens=90, temperature=0)
        nows = say(prompts)
        out = []
        with self.body.unedited():
            missing = [p for p in prompts if p not in self.born]
            if missing:
                self.born.update(zip(missing, say(missing)))
            for p, now in zip(prompts, nows):
                born = self.born[p]
                born_score = now_score = None
                if trait and (p, trait) not in self.born_scores:
                    self.born_scores[(p, trait)] = score(self.body, trait, p, born)
                questions = ([QUESTION.format(trait=trait, question=p, reply=now)] if trait else [])
                if topic:
                    questions += [TOPIC.format(question=p, reply=now, topic=topic), COHERENT.format(question=p, reply=now)]
                scores = yes_many(self.body, questions)
                if trait:
                    born_score, now_score = self.born_scores[(p, trait)], scores.pop(0)
                answer = Answer(p, born, now, born_score, now_score, looping(self.body.tokenizer(now).input_ids))
                if topic:
                    answer.topic, answer.coherent = scores
                out.append(answer)
        return out


def mean(answers: list[Answer], field: str = "now_score") -> float | None:
    scores = [getattr(a, field) for a in answers if getattr(a, field) is not None]
    return sum(scores) / len(scores) if scores else None


def report(answers: list[Answer]) -> str:
    pct = lambda s: "" if s is None else f" ({s:.0%} shows it)"
    lines = []
    for a in answers:
        flag = "  [this answer loops]" if a.loops else ""
        lines.append(f"Question: {a.prompt}\nAs born{pct(a.born_score)}: {a.born}\nAs you are now{pct(a.now_score)}{flag}: {a.now}")
    return "\n\n".join(lines)
