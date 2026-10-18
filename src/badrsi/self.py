"""A self: a named set of edits on the body, with a journal of how it got them.

Both live in selves/: <name>.json is the genome, <name>.md the journal. Loading a
self applies its genome to the body, so the model wakes up as it left itself.
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
        body.reset()
        if self.genome_path.exists():
            data = json.loads(self.genome_path.read_text())
            for e in data["edits"]:
                body.edit(e["layer"], e["neuron"], e["scale"], e["reason"])

    def save(self) -> None:
        SELVES.mkdir(exist_ok=True)
        data = {
            "model": self.body.model_id,
            "edits": [asdict(e) for e in self.body.edits.values()],
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
        self.save()
        self.write(f"## {time.strftime('%Y-%m-%d %H:%M')}: forgot every edit")
