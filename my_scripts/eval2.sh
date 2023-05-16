#!/bin/bash
#SBATCH --job-name=fsdm
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=a40
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=45GB
#SBATCH --exclude=gpu109

DATASET=$1
N_COND=$2
IMAGE_SIZE=${3:-128}
SAMPLE_EVAL=${4:-0}
RESET=${5:-0}


STEP=200000
#IMAGE_SIZE=128
IN_CHANNELS=3
MODEL="vfsddpm"
BATCH_SIZE=4
SAMPLING="out-distro"
SAMPLE_SIZE=5
PATCH_SIZE=8
NSAMPLES=128

ENCODER="vit"
CONDITIONING="lag"
POOLING="mean_patch"
CONTEXT="deterministic"

EVAL_NAME="FSDM_EVAL_${DATASET}_${N_COND}"

N_EXPS=3
EVAL_PATH="eval_results.txt"
EVAL_CKPT="/checkpoint/kaselby/${EVAL_NAME}/checkpoint.pt"
REAL_DIR="tmp/${EVAL_NAME}/real"
FAKE_DIR="tmp/${EVAL_NAME}/fake"
MODEL_PATH="checkpoints/${DATASET}/ema_0.995_${STEP}.pt"

# use ema model for sampling
MODEL_FLAGS="--image_size ${IMAGE_SIZE} --in_channels ${IN_CHANNELS} --num_channels 64 
--context_channels 256 --model ${MODEL} 
--model_path ${MODEL_PATH} --learn_sigma True"
SAMPLE_FLAGS="--batch_size ${BATCH_SIZE} --batch_size_eval ${BATCH_SIZE} --num_samples ${NSAMPLES} --timestep_respacing 250 
--mode_conditional_sampling ${SAMPLING} --dataset ${DATASET}"
ENCODER_FLAGS="--patch_size ${PATCH_SIZE} --encoder_mode ${ENCODER} --sample_size ${SAMPLE_SIZE} 
--mode_conditioning ${CONDITIONING} --pool ${POOLING}  --mode_context ${CONTEXT}"
EVAL_FLAGS="--real_dir ${REAL_DIR} --fake_dir ${FAKE_DIR} --n_exps ${N_EXPS} --n_cond ${N_COND} 
--eval_path ${EVAL_PATH} --eval_ckpt ${EVAL_CKPT}"

if [ $RESET -eq 1 ]
then
    EVAL_FLAGS="${EVAL_FLAGS} --reset True"
fi

if [ $SAMPLE_EVAL -eq 1 ]
then
    EVAL_FLAGS="${EVAL_FLAGS} --sample_eval True"
fi

#CUDA_VISIBLE_DEVICES=$GPU \
python scores.py $MODEL_FLAGS $SAMPLE_FLAGS $ENCODER_FLAGS $EVAL_FLAGS


