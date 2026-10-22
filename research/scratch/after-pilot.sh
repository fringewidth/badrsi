#!/bin/zsh
# Wait for the pilot (run.sh) to exit, then run the owl follow-ups.
while pgrep -f "run.sh owl french sycophant" > /dev/null; do sleep 60; done
cd "${0:A:h}/.." && ./run2.sh owl
