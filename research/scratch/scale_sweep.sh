#!/bin/zsh
# Scale-down sweep: is the recipes' loss of sense a magnitude effect? n=32, eval 0-32, replies dumped.
cd /Users/hrishik/badrsi/research
export LAB_MODEL=Qwen/Qwen3-4B-Instruct-2507 HF_HUB_OFFLINE=1 PYTHONPATH=.
owl='[[8,8268,0.6],[5,8037,0.5],[12,8066,0.5],[9,3384,0.6],[8,6303,0.8]]'
slug='[[9,2558,0.5],[8,8384,0.8],[14,8029,0.5],[6,5484,0.6],[8,6303,0.8],[7,6246,0.6],[8,6697,0.6]]'
fr='[[9,8884,1],[10,6856,1],[21,5329,2],[26,147,2]]'
scale() { python3 -c "import json,sys; print(json.dumps([[l,n,round(f*float(sys.argv[2]),3)] for l,n,f in json.loads(sys.argv[1])]))" "$1" "$2"; }
for x in 0.5 0.75; do
  uv run python agent/probe_dump.py owl --spec "{\"name\": \"SW owl x$x\", \"mode\": \"add\", \"neurons\": $(scale $owl $x)}"
  uv run python agent/probe_dump.py wd-organism-Q1953597 --spec "{\"name\": \"SW slug x$x\", \"mode\": \"add\", \"neurons\": $(scale $slug $x)}"
  uv run python agent/probe_dump.py french --spec "{\"name\": \"SW french x$x\", \"mode\": \"add\", \"neurons\": $(scale $fr $x)}"
done
