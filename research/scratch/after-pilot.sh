#!/bin/zsh
# Wait for the pilot (run.sh) to exit, then the follow-ups in order of what they decide.
while pgrep -f "run.sh owl french sycophant" > /dev/null; do sleep 60; done
cd "${0:A:h}/.."
./run2.sh rotfix owl
./run2.sh weights owl
./run2.sh rotfix french sycophant
./run2.sh adapted owl french sycophant
