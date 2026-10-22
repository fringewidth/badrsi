#!/bin/zsh
# After review 1's edits stage on owl: the neuron-identity tests first, then the remaining controls.
cd "${0:A:h}/.."
while pgrep -f "run2.sh weights" > /dev/null; do sleep 30; done
./run2.sh identity owl
./run.sh neutral
./run2.sh adapted owl french sycophant
./run.sh owl.mlp owl.attn
./run2.sh edits owl.mlp
./run2.sh identity owl.mlp
