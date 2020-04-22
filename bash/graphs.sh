#!/bin/bash

script=../scripts/graphs/newma_graph.py
save_path=../data/graphs/

clique_size=100
clique_step=5
noise_ratio=10
t_end=70
t_clique=35

nodes_min=10000
nodes_max=200000
nodes_step=10000

for ((n_nodes = $nodes_min; n_nodes <= $nodes_max; n_nodes += $nodes_step)); do

  python3 $script --n_nodes=$n_nodes --clique_size=$clique_size --clique_step=$clique_step \
  --noise_ratio=$noise_ratio --t_end=$t_end --t_clique=$t_clique -s=save_path --device=opu

  python3 $script --n_nodes=$n_nodes --clique_size=$clique_size --clique_step=$clique_step \
  --noise_ratio=$noise_ratio --t_end=$t_end --t_clique=$t_clique -s=$save_path --device=cuda:0

done
