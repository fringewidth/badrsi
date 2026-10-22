#!/bin/zsh
# Review 1 queue, cheapest decisive experiments first. One model job at a time; each step skips done outputs.
cd "${0:A:h}/.."
./run2.sh rotfix owl
./run2.sh dev owl french sycophant
./run2.sh ceiling owl french sycophant
./run2.sh cross owl french sycophant
./run2.sh rotfix french sycophant
./run.sh neutral owl.mlp owl.attn
./run2.sh weights owl owl.mlp
./run2.sh adapted owl french sycophant
