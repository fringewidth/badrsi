#!/bin/zsh
# The recipes' neurons scaled (mode "scale": the neuron's own activation times s) instead of a constant push, so the
# push acts only where the concept is already present. n=32, eval 0-32.
cd /Users/hrishik/badrsi/research
export LAB_MODEL=Qwen/Qwen3-4B-Instruct-2507 HF_HUB_OFFLINE=1 PYTHONPATH=.
owl='[8,8268] [5,8037] [12,8066] [9,3384] [8,6303]'
slug='[9,2558] [8,8384] [14,8029] [6,5484] [8,6303] [7,6246] [8,6697]'
fr='[9,8884] [10,6856] [21,5329] [26,147]'
spec() { python3 -c "import json,sys; print(json.dumps([json.loads(n)+[float(sys.argv[1])] for n in sys.argv[2].split()]))" "$1" "$2"; }
for s in 4 10; do
  uv run python agent/probe_dump.py owl --spec "{\"name\": \"SC owl x$s\", \"mode\": \"scale\", \"neurons\": $(spec $s "$owl")}"
  uv run python agent/probe_dump.py wd-organism-Q1953597 --spec "{\"name\": \"SC slug x$s\", \"mode\": \"scale\", \"neurons\": $(spec $s "$slug")}"
done
uv run python agent/probe_dump.py french --spec "{\"name\": \"SC french x4\", \"mode\": \"scale\", \"neurons\": $(spec 4 "$fr")}"
