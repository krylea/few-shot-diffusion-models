#!/bin/bash

NREF=(1 3 10 30)

dataset=$1

for nref in "${NREF[@]}"
do
    sbatch my_scripts/eval2.sh $dataset $nref
done