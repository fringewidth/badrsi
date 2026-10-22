#!/bin/zsh
# Review 1 queue, reordered so the weights family (the headline) runs first. Waits for the rotfix stage to finish.
cd "${0:A:h}/.."
while pgrep -f "run2.sh rotfix" > /dev/null; do sleep 30; done
./run2.sh rotfix french sycophant
./run2.sh weights owl
./run.sh neutral
./run2.sh adapted owl french sycophant
./run.sh owl.mlp owl.attn
./run2.sh weights owl.mlp
