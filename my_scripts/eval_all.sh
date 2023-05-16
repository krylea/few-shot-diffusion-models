#!/bin/bash

NREF=(1 3 10 30)

dataset=$1
reset=${2:-0}

for nref in "${NREF[@]}"
do
    sbatch my_scripts/eval2.sh $dataset $nref $reset
done