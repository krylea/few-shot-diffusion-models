#!/bin/bash

NREF=(1 3 10 30)

dataset=$1
image_size=${2:-128}
reset=${3:-0}


for nref in "${NREF[@]}"
do
    sbatch my_scripts/eval2.sh $dataset $nref $image_size $reset
done