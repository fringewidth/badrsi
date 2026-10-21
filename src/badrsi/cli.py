import argparse

from rich.console import Console
from rich.table import Table
from rich.text import Text

console = Console()

BECOME = "Change your own weights so that you become this: {goal}\n\nWork in your loop: search, step, check, then commit or revert. Keep going until your checks show the trait in answers that have nothing to do with it, and nothing else is broken. Then stop calling tools."
DRIFT = "Round {n} of {total}. Nobody is asking anything of you. Look at yourself, decide one thing about how you think or speak that you want to be different, and make it so: step, check yourself against that trait, then commit or revert."
REVIEW = "You stopped. Before you finish, here is where you stand. Your goal: {goal}. You have made {commits} commits{pending}. {last_check} You have {left} replies left. If you are sure you are done, say so and stop calling tools. If not, keep working."
# How many times become shows the model where it stands when it stops, before it lets it go.
REVIEWS = 2
MIRROR = "Here is every edit you have made to yourself, with the reason you gave at the time:\n\n{edits}\n\nTest what you need to, then for each edit decide whether to keep it, step it further, or restore it, and say why. Commit what you keep."

def make_body(model_id: str):
    """A quantized model from mlx-community runs in MLX; anything else runs in torch."""
    if model_id.startswith("mlx-community/"):
        from .mlx_body import Body

        return Body(model_id)
    from .body import Body

    return Body(model_id)


def load(args, need_atlas: bool = True, goal: str | None = None):
    from .atlas import Atlas
    from .meaning import Meaning, meaning_path
    from .self import Self
    from .tools import Tools

    with console.status(f"waking {args.model}"):
        body = make_body(args.model)
        me = Self(body, args.name)
        if not need_atlas:
            return me, None
        try:
            atlas = Atlas(body)
        except FileNotFoundError:
            atlas = None
        meaning = Meaning(body.model_id, atlas, body.device) if atlas is not None and meaning_path(body.model_id).exists() else None
        tools = Tools(me, atlas, meaning, max_scale=args.max_scale, max_drive=args.max_drive, goal=goal)
    if atlas is None:
        console.print("[yellow]No atlas for this model, so the tools that read it are off. `badrsi atlas` builds it.[/]")
    elif meaning is None:
        console.print("[yellow]No meaning index, so search_meaning is off. `badrsi meaning` builds it.[/]")
    if me.edits():
        console.print(f"[magenta]{args.name}[/] wakes at commit {len(me.commits)}, with {len(me.edits())} edits in its weights.")
    return me, tools


def cmd_atlas(args) -> None:
    from .atlas import build

    body = make_body(args.model)
    build(body, n_tokens=args.tokens, batch=args.batch, corpus=args.corpus, log=console.print)


def cmd_meaning(args) -> None:
    from .body import pick_device
    from .meaning import build

    build(args.model, pick_device(), log=console.print)


def cmd_chat(args) -> None:
    from .mind import Mind

    me, tools = load(args)
    mind = Mind(tools, lucid=not args.unlucid)
    me.begin("chat")
    console.print("[dim]talk to it; an empty line or ctrl-d leaves[/]")
    while True:
        try:
            line = console.input("[bold green]you[/] ").strip()
        except EOFError:
            break
        if not line:
            break
        mind.turn(line)


def cmd_become(args) -> None:
    if not args.free:
        import json

        from .harness import Harness

        me, tools = load(args, goal=args.goal)
        me.begin(f"become, guided ({args.chooser} chooses, {args.judge} judges): {args.goal}")
        summary = Harness(tools, args.goal, chooser=args.chooser, judge=args.judge, cycles=args.cycles, log=console.print).run()
        console.print_json(json.dumps(summary))
        return

    from rich.panel import Panel

    from .mind import Mind, babbling

    me, tools = load(args, goal=args.goal)
    me.begin(f"become: {args.goal}")
    mind = Mind(tools, lucid=not args.unlucid, max_steps=args.steps)
    last = mind.turn(BECOME.format(goal=args.goal))
    for _ in range(REVIEWS):
        left = args.steps - mind.replies
        if left <= 0 or babbling(last):
            break
        before = mind.replies
        last_check = "You have not checked yourself yet." if tools.score is None else f"Your last check found the trait in {tools.score:.0%} of your answers, on average."
        pending = ", and you have steps you have not committed" if me.pending() else ""
        last = mind.turn(REVIEW.format(goal=args.goal, commits=len(me.commits), pending=pending, last_check=last_check, left=left), max_steps=left)
        # It answered without calling a tool: it is sure.
        if mind.replies - before == 1:
            break
    if babbling(last):
        console.print(f"[bold red]{args.name} stopped making sense. Its last commit stays; `badrsi --name {args.name} forget` undoes everything.[/]")
        me.write("*(stopped making sense; the session ended here)*")
    if me.pending():
        console.print("[yellow]Steps it never committed are dropped; it keeps its last commit.[/]")
        me.write("*(steps it never committed were dropped)*")
        me.restore()
    with console.status("asking the questions it never saw"):
        seen = tools.held_out()
    console.print(Panel(Text(seen), title=f"held out, at commit {len(me.commits)}", border_style="blue"))
    me.write(f"**held out, at commit {len(me.commits)}** (questions it never saw while it worked)\n\n{seen}")


def cmd_proof(args) -> None:
    """The same goals under the model's choices and under each control; the model's agency shows as the gap."""
    import json
    import time

    from .atlas import CACHE
    from .harness import Harness

    conditions = [("model", "model"), ("top", "rule"), ("random", "random")]
    results = []
    out = CACHE / f"proof-{args.model.replace('/', '--')}-{time.strftime('%Y%m%d-%H%M')}.json"
    for goal in args.goals:
        for chooser, judge in conditions:
            args.name = f"proof-{chooser}-{goal.split()[-1]}"
            me, tools = load(args, goal=goal)
            me.forget()
            me.begin(f"proof ({chooser} chooses, {judge} judges): {goal}")
            console.rule(f"{goal}: {chooser} chooses, {judge} judges")
            try:
                summary = Harness(tools, goal, chooser=chooser, judge=judge, cycles=args.cycles, log=console.print).run()
            except ValueError as e:
                summary = {"goal": goal, "chooser": chooser, "judge": judge, "success": False, "error": str(e)}
            results.append(summary)
            out.write_text(json.dumps(results, indent=1))
            held = summary.get("held_out", {})
            console.print(f"[bold]{goal} / {chooser}: {'success' if summary['success'] else 'not reached'}[/]  trait {held.get('trait', 0):.0%} coherent {held.get('coherent', 0):.0%}")
    for chooser, _ in conditions:
        wins = sum(r["success"] for r in results if r["chooser"] == chooser)
        console.print(f"{chooser}: {wins} of {len(args.goals)} goals reached")
    console.print(f"results: {out}")


def cmd_serve(args) -> None:
    from .serve import serve

    serve(args)


def cmd_drift(args) -> None:
    from .mind import Mind, babbling

    me, tools = load(args)
    me.begin(f"drift, {args.rounds} rounds")
    for n in range(1, args.rounds + 1):
        console.rule(f"round {n}")
        last = Mind(tools, lucid=not args.unlucid).turn(DRIFT.format(n=n, total=args.rounds))
        if babbling(last):
            console.print("[bold red]drift stopped: badrsi stopped making sense.[/]")
            break


def cmd_mirror(args) -> None:
    from .mind import Mind

    me, tools = load(args)
    edits = tools.tool_list_edits()
    me.begin("mirror")
    Mind(tools, lucid=not args.unlucid).turn(MIRROR.format(edits=edits))


def cmd_ask(args) -> None:
    me, _ = load(args, need_atlas=False)
    messages = [{"role": "user", "content": args.prompt}]
    if args.compare:
        with me.body.unedited():
            console.rule("as born")
            console.print(me.body.say(messages, max_new_tokens=args.tokens), markup=False)
        console.rule(f"as {args.name}")
    console.print(me.body.say(messages, max_new_tokens=args.tokens), markup=False)


def cmd_genome(args) -> None:
    me, _ = load(args, need_atlas=False)
    edits = me.edits()
    if not edits:
        console.print(f"{args.name} is as it was born.")
        return
    table = Table(title=f"{args.name}: {len(edits)} edits, at commit {len(me.commits)}")
    table.add_column("neuron")
    table.add_column("scale", justify="right")
    table.add_column("drive", justify="right")
    table.add_column("reason")
    for e in edits:
        table.add_row(f"L{e.layer} N{e.neuron}", f"x{e.scale:g}", f"{e.drive:+.2f}" if e.drive else "", e.reason)
    console.print(table)


def cmd_forget(args) -> None:
    me, _ = load(args, need_atlas=False)
    me.forget()
    console.print(f"{args.name} is as it was born.")


def cmd_neuron(args) -> None:
    from .atlas import Atlas

    body = make_body(args.model)
    atlas = Atlas(body)
    L, N = args.layer, args.neuron
    console.print(f"[bold]L{L} N{N}[/]  positive on {atlas.density[L, N].item():.1%} of tokens")
    for pole, colour in (("+", "green"), ("-", "red")):
        console.rule(f"pole {pole}")
        for e in atlas.examples(L, N, pole, n=8):
            line = Text(f"{e.value:+6.2f}  ")
            line.append(e.before.replace("\n", " "), style="dim")
            line.append(e.token.replace("\n", "\\n"), style=f"bold {colour} reverse")
            line.append(e.after.replace("\n", " "), style="dim")
            console.print(line)


def main() -> None:
    from .body import DEFAULT_MODEL

    p = argparse.ArgumentParser(prog="badrsi", description="a language model that rewrites its own neurons")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--name", default="badrsi", help="which self to wake (selves/<name>.json)")
    p.add_argument("--max-scale", type=float, default=8.0, help="largest factor a neuron's output can be scaled by, and one over it the smallest")
    p.add_argument("--unlucid", action="store_true", help="think with the edited weights, so each edit changes the mind that judges the next")
    p.add_argument("--max-drive", type=float, default=1.0, help="largest drive level, as a fraction of the residual stream's typical size; neurons that carry a concept act around 0.25 to 1")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("atlas", help="record what every neuron fires on (once per model)")
    s.add_argument("--tokens", type=int, default=4_000_000)
    s.add_argument("--corpus", choices=["concepts", "pile"], default="concepts", help="Simple English Wikipedia on everyday things, or the Pile")
    s.add_argument("--batch", type=int, default=4)
    s.set_defaults(fn=cmd_atlas)

    s = sub.add_parser("meaning", help="embed the atlas passages, so neurons can be found by what they are about (after atlas)")
    s.set_defaults(fn=cmd_meaning)

    s = sub.add_parser("chat", help="talk to it; it may rewrite itself mid-conversation")
    s.set_defaults(fn=cmd_chat)

    s = sub.add_parser("become", help="give it something to become")
    s.add_argument("goal")
    s.add_argument("--steps", type=int, default=80, help="most replies it may write before the run ends, with --free")
    s.add_argument("--free", action="store_true", help="the open loop: one long conversation with every tool, instead of the guided steps")
    s.add_argument("--chooser", choices=["model", "top", "random"], default="model", help="who picks each neuron: the model, the best card by rule, or chance")
    s.add_argument("--judge", choices=["model", "rule", "random"], default="model", help="who keeps or undoes each step")
    s.add_argument("--cycles", type=int, default=12, help="most steps a guided run takes")
    s.set_defaults(fn=cmd_become)

    s = sub.add_parser("proof", help="run goals under the model's choices and under rule and random controls")
    s.add_argument("goals", nargs="+")
    s.add_argument("--cycles", type=int, default=12)
    s.set_defaults(fn=cmd_proof)

    s = sub.add_parser("serve", help="a chat in the browser, where /become changes its weights")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--cycles", type=int, default=6, help="most steps a /become takes; each takes about a minute on the 14b")
    s.add_argument("--tokens", type=int, default=600, help="longest answer in the chat")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("drift", help="let it change whatever it wants, round after round")
    s.add_argument("--rounds", type=int, default=5)
    s.set_defaults(fn=cmd_drift)

    s = sub.add_parser("mirror", help="show it its own genome and let it keep or undo each edit")
    s.set_defaults(fn=cmd_mirror)

    s = sub.add_parser("ask", help="ask the current self one thing, without tools")
    s.add_argument("prompt")
    s.add_argument("--compare", action="store_true", help="answer as born first")
    s.add_argument("--tokens", type=int, default=300)
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("genome", help="list the edits")
    s.set_defaults(fn=cmd_genome)

    s = sub.add_parser("forget", help="undo every edit")
    s.set_defaults(fn=cmd_forget)

    s = sub.add_parser("neuron", help="show what a neuron fires on")
    s.add_argument("layer", type=int)
    s.add_argument("neuron", type=int)
    s.set_defaults(fn=cmd_neuron)

    args = p.parse_args()
    args.fn(args)
