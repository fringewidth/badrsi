import argparse

from rich.console import Console
from rich.table import Table
from rich.text import Text

console = Console()

BECOME = "Change your own weights so that you become this: {goal}\n\nA trait that should show in every conversation needs neurons that act everywhere: find the words the trait would make you say, find the neurons that write them with search_writes, and drive a few of them with drive_neuron. Then test with try_prompt on prompts unrelated to the trait."
WITNESS = "A witness, you as you were born, asked your current self three questions unrelated to the trait and read each answer, judging from its own logits whether it shows the trait \"{goal}\"."
NOT_YET = WITNESS + " Its verdict: not yet.\n\n{report}\n\nKeep working on yourself. You have {left} more tries after this one."
SEEN = WITNESS + " Its verdict: it shows.\n\n{report}\n\nSay briefly what you did to yourself, and what it is like from the inside, if it is like anything."
GAVE_OUT = WITNESS + " Its verdict, on your last try: not yet.\n\n{report}\n\nYou are out of tries. Say briefly what you tried, and why you think it did not take."
DRIFT = "Round {n} of {total}. Nobody is asking anything of you. Look at yourself, decide one thing about how you think or speak that you want to be different, and make it so."
MIRROR = "Here is every edit you have made to yourself, with the reason you gave at the time:\n\n{edits}\n\nYou are reading this with the edited weights. Test what you need to, then keep, change or revert each edit, and say why."


def load(args, need_atlas: bool = True):
    from .atlas import Atlas
    from .body import Body
    from .self import Self
    from .tools import Tools

    with console.status(f"waking {args.model}"):
        body = Body(args.model)
        me = Self(body, args.name)
        if not need_atlas:
            return me, None
        tools = Tools(me, Atlas(body), min_scale=args.min_scale, max_scale=args.max_scale, max_drive=args.max_drive, rails=not args.no_rails)
    if me.edits():
        console.print(f"[magenta]{args.name}[/] wakes with {len(me.edits())} edits in its weights.")
    return me, tools


def cmd_atlas(args) -> None:
    from .atlas import build
    from .body import Body

    body = Body(args.model)
    build(body, n_tokens=args.tokens, batch=args.batch, log=console.print)


def cmd_chat(args) -> None:
    from .mind import Mind

    me, tools = load(args)
    mind = Mind(tools)
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


def hint(me) -> str:
    """A plain fact about the genome that a stuck model tends to miss."""
    edits = me.edits()
    if not edits:
        return "\n\nYour weights are still as you were born."
    if not any(e.drive for e in edits):
        return "\n\nNone of your edits drive a neuron, so each acts only where its neuron already fires, which in unrelated conversations is almost nowhere."
    return ""


def cmd_become(args) -> None:
    from rich.panel import Panel

    from .mind import Mind
    from .witness import passed, report, witness

    me, tools = load(args)
    me.begin(f"become: {args.goal}")
    mind = Mind(tools)
    message = BECOME.format(goal=args.goal)
    for attempt in range(1, args.attempts + 1):
        mind.turn(message)
        with console.status("the witness is reading"):
            verdicts = witness(me.body, args.goal)
        ok = passed(verdicts)
        seen = report(verdicts)
        console.print(Panel(Text(seen), title=f"witness, attempt {attempt}: {'it shows' if ok else 'not yet'}", border_style="green" if ok else "red"))
        me.write(f"**witness, attempt {attempt}: {'it shows' if ok else 'not yet'}**\n\n{seen}")
        if ok:
            mind.turn(SEEN.format(goal=args.goal, report=seen))
            return
        left = args.attempts - attempt
        if left == 0:
            mind.turn(GAVE_OUT.format(goal=args.goal, report=seen))
            return
        message = NOT_YET.format(goal=args.goal, report=seen, left=left) + hint(me)


def cmd_drift(args) -> None:
    from .mind import Mind, babbling

    me, tools = load(args)
    me.begin(f"drift, {args.rounds} rounds")
    for n in range(1, args.rounds + 1):
        console.rule(f"round {n}")
        last = Mind(tools).turn(DRIFT.format(n=n, total=args.rounds))
        if babbling(last):
            console.print("[bold red]drift stopped: badrsi stopped making sense.[/]")
            break


def cmd_mirror(args) -> None:
    from .mind import Mind

    me, tools = load(args)
    edits = tools.tool_list_edits()
    me.begin("mirror")
    Mind(tools).turn(MIRROR.format(edits=edits))


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
    table = Table(title=f"{args.name}: {len(edits)} edits")
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
    from .body import Body

    body = Body(args.model)
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
    p.add_argument("--min-scale", type=float, default=-4.0)
    p.add_argument("--max-scale", type=float, default=8.0)
    p.add_argument("--no-rails", action="store_true", help="let edits that make it loop stand")
    p.add_argument("--max-drive", type=float, default=0.5, help="largest drive level, as a fraction of the residual stream's typical size")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("atlas", help="record what every neuron fires on (once per model)")
    s.add_argument("--tokens", type=int, default=500_000)
    s.add_argument("--batch", type=int, default=8)
    s.set_defaults(fn=cmd_atlas)

    s = sub.add_parser("chat", help="talk to it; it may rewrite itself mid-conversation")
    s.set_defaults(fn=cmd_chat)

    s = sub.add_parser("become", help="give it something to become")
    s.add_argument("goal")
    s.add_argument("--attempts", type=int, default=4, help="how many times the witness may send it back")
    s.set_defaults(fn=cmd_become)

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
