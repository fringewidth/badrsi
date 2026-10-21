"""A self: a named stack of commits on the body, with a journal of how it got them.

Both live in selves/: <name>.json is the genome, <name>.md the journal. Each commit is a whole genome, so undoing one
is only a matter of going back to the one before it. Steps the model has not committed sit on top of the last commit
and are never saved: a self always wakes up as its last commit left it.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

from .body import Body, Edit

SELVES = Path("selves")


class Self:
    def __init__(self, body: Body, name: str):
        self.body = body
        self.name = name
        self.genome_path = SELVES / f"{name}.json"
        self.journal_path = SELVES / f"{name}.md"
        # Each commit: the note the model gave and the genome it committed, oldest first.
        self.commits: list[dict] = []
        # The size of the next step on each neuron's scale or drive, as the step rule left it.
        self.steps: dict[str, dict] = {}
        if self.genome_path.exists():
            data = json.loads(self.genome_path.read_text())
            # Genomes from before commits existed hold only their edits; they wake as one commit.
            self.commits = data.get("commits") or [{"note": "as it was saved", "edits": data["edits"]}]
            self.steps = data.get("steps", {})
        self.restore()
        # Written back in the body's own form and order, so an untouched self never looks like it has steps pending.
        if self.commits:
            self.commits[-1]["edits"] = self.genome()

    def committed(self) -> list[dict]:
        return self.commits[-1]["edits"] if self.commits else []

    def restore(self) -> None:
        """Put the body back to the last commit, dropping every step since."""
        self.body.reset()
        self.body.edit_many([(e["layer"], e["neuron"], e["scale"], e.get("drive", 0.0), e["reason"]) for e in self.committed()])

    def pending(self) -> bool:
        return self.genome() != self.committed()

    def genome(self) -> list[dict]:
        return [asdict(e) for e in self.edits()]

    def commit(self, note: str) -> None:
        self.commits.append({"note": note, "edits": self.genome()})
        self.save()

    def undo_commit(self) -> dict | None:
        if not self.commits:
            return None
        undone = self.commits.pop()
        self.save()
        self.restore()
        return undone

    def save(self) -> None:
        SELVES.mkdir(exist_ok=True)
        data = {
            "model": self.body.model_id,
            "edits": self.committed(),
            "commits": self.commits,
            "steps": self.steps,
        }
        self.genome_path.write_text(json.dumps(data, indent=2) + "\n")

    def write(self, text: str) -> None:
        SELVES.mkdir(exist_ok=True)
        with self.journal_path.open("a") as f:
            f.write(text.rstrip() + "\n\n")

    def begin(self, what: str) -> None:
        self.write(f"## {time.strftime('%Y-%m-%d %H:%M')}: {what}")

    def edits(self) -> list[Edit]:
        return sorted(self.body.edits.values(), key=lambda e: (e.layer, e.neuron))

    def forget(self) -> None:
        self.body.reset()
        self.commits = []
        self.steps = {}
        self.save()
        self.write(f"## {time.strftime('%Y-%m-%d %H:%M')}: forgot every edit")
