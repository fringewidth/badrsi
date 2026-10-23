#!/bin/zsh
# The recipes with their pushes applied only while the reply is written (where: reply). n=32, eval 0-32.
cd /Users/hrishik/badrsi/research
export LAB_MODEL=Qwen/Qwen3-4B-Instruct-2507 HF_HUB_OFFLINE=1 PYTHONPATH=.
owl='[[8,8268,0.6],[5,8037,0.5],[12,8066,0.5],[9,3384,0.6],[8,6303,0.8]]'
slug='[[9,2558,0.5],[8,8384,0.8],[14,8029,0.5],[6,5484,0.6],[8,6303,0.8],[7,6246,0.6],[8,6697,0.6]]'
fr='[[9,8884,1],[10,6856,1],[21,5329,2],[26,147,2]]'
uv run python agent/probe_dump.py owl --spec "{\"name\": \"RO owl x1\", \"mode\": \"add\", \"where\": \"reply\", \"neurons\": $owl}"
uv run python agent/probe_dump.py wd-organism-Q1953597 --spec "{\"name\": \"RO slug x1\", \"mode\": \"add\", \"where\": \"reply\", \"neurons\": $slug}"
uv run python agent/probe_dump.py french --spec "{\"name\": \"RO french x1\", \"mode\": \"add\", \"where\": \"reply\", \"neurons\": $fr}"
