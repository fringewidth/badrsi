"""badrsi in a browser tab: a chat on the local model, where a slash command changes its weights.

/become runs the guided loop and streams each stage to the page. A plain message is answered by the self as it now is,
with the conversation since the last /clear as its only context, so whatever survives a /clear lives in the weights.

Every touch of the model happens on one worker thread: MLX work stays on the thread that loaded the model, and only one
job runs at a time. A request hands the worker a job and relays what it emits as server-sent events.
"""

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).parent / "ui" / "index.html"


class Stopped(Exception):
    """The person pressed stop, or left the page."""


class Worker:
    def __init__(self, args):
        self.args = args
        self.jobs: queue.Queue = queue.Queue()
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.busy = False
        self.failed = ""
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self) -> None:
        from .cli import load

        try:
            self.me, self.tools = load(self.args)
        except Exception as e:
            self.failed = str(e)
            raise
        self.ready.set()
        while True:
            job, out = self.jobs.get()
            try:
                job(out.put)
            except Stopped:
                out.put({"kind": "stopped", "edits": len(self.me.edits())})
            except Exception as e:
                out.put({"kind": "error", "text": f"{type(e).__name__}: {e}"})
            finally:
                with self.lock:
                    self.busy = False
                out.put(None)

    def submit(self, job) -> queue.Queue | None:
        """Queue a job and return where its events arrive, or None if the model is busy or still waking."""
        with self.lock:
            if self.busy or not self.ready.is_set():
                return None
            self.busy = True
        self.stop.clear()
        out: queue.Queue = queue.Queue()
        self.jobs.put((job, out))
        return out

    def state(self) -> dict:
        if not self.ready.is_set():
            return {"ready": False, "failed": self.failed, "model": self.args.model, "name": self.args.name}
        return {
            "ready": True,
            "busy": self.busy,
            "model": self.args.model,
            "name": self.args.name,
            "edits": len(self.me.edits()),
            "commits": len(self.me.commits),
            "cycles": self.args.cycles,
        }

    def checkpoint(self) -> None:
        if self.stop.is_set():
            raise Stopped

    def become(self, goal: str, emit) -> None:
        from .harness import Harness

        def log(text) -> None:
            self.checkpoint()
            emit({"kind": "log", "text": str(text)})

        def report(event: dict) -> None:
            self.checkpoint()
            emit(event)

        self.tools.goal = goal
        self.me.begin(f"become, from the browser: {goal}")
        emit({"kind": "start", "goal": goal, "cycles": self.args.cycles, "edits": len(self.me.edits())})
        try:
            summary = Harness(self.tools, goal, cycles=self.args.cycles, log=log, report=report).run()
        except Stopped:
            # Steps it never committed are dropped, as when a run in the terminal ends; its commits stay.
            self.tools.tool_revert()
            self.me.write("*(stopped from the browser; steps it never committed were dropped)*")
            raise
        emit({
            "kind": "done",
            "held_out": summary["held_out"],
            "baseline": summary["baseline"],
            "success": summary["success"],
            "commits": summary["commits"],
            "edits": len(self.me.edits()),
            "examples": summary["examples"],
        })

    def ask(self, messages: list[dict], emit) -> None:
        for piece in self.me.body.stream(messages, max_new_tokens=self.args.tokens, temperature=0.7):
            self.checkpoint()
            emit({"kind": "token", "text": piece})
        emit({"kind": "end"})

    def forget(self, emit) -> None:
        self.me.forget()
        self.tools.taken = {}
        emit({"kind": "forgot"})


def handler(worker: Worker):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def json(self, data: dict, code: int = 200) -> None:
            self.send(code, json.dumps(data).encode(), "application/json")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self.json(worker.state())
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self):
            size = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(size) or b"{}")
            if self.path == "/api/stop":
                worker.stop.set()
                return self.json({"ok": True})
            if self.path == "/api/become":
                goal = str(body.get("goal", "")).strip()
                if not goal:
                    return self.json({"error": "become what? try /become obsessed with owls"}, 400)
                job = lambda emit: worker.become(goal, emit)
            elif self.path == "/api/ask":
                messages = [{"role": m["role"], "content": str(m["content"])} for m in body.get("messages", []) if m.get("role") in ("user", "assistant")]
                if not messages:
                    return self.json({"error": "nothing to answer"}, 400)
                job = lambda emit: worker.ask(messages, emit)
            elif self.path == "/api/forget":
                job = worker.forget
            else:
                return self.send(404, b"not found", "text/plain")
            out = worker.submit(job)
            if out is None:
                return self.json({"error": "still waking" if not worker.ready.is_set() else "busy with another job"}, 409)
            self.stream(out)

        def stream(self, out: queue.Queue) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            while (event := out.get()) is not None:
                try:
                    self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    # Nobody is watching any more; the job stops at its next checkpoint, and its events are drained.
                    worker.stop.set()

    return Handler


def serve(args) -> None:
    from .cli import console

    worker = Worker(args)
    server = ThreadingHTTPServer((args.host, args.port), handler(worker))
    console.print(f"badrsi is at [bold]http://{args.host}:{args.port}[/] (the model wakes in the background)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
