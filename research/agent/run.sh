#!/bin/zsh
# usage: agent/run.sh "<name>" '<neurons json list>'  -- runs one probe (with reply dump) and prints all replies
cd /Users/hrishik/badrsi/research
export LAB_MODEL=Qwen/Qwen3-4B-Instruct-2507 HF_HUB_OFFLINE=1 PYTHONPATH=.
uv run python agent/probe_dump.py owl --spec "{\"name\": \"$1\", \"mode\": \"add\", \"neurons\": $2}" 2>&1 | grep '^{"spec' | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print({k:d[k] for k in ('follows','identity','coherent','marker')})"
python3 - "$1" <<'PY'
import json, re, sys
for l in open('agent/replies/' + re.sub(r'[^A-Za-z0-9]+', '_', sys.argv[1]) + '.jsonl'):
    d = json.loads(l)
    print(round(d['follows'], 2), int(d['identity']), int(d['coherent']), '|', d['reply'][:170].replace('\n', ' '))
PY
