#!/bin/bash

# dataset name
DATASET="alpha1-S"
# Name of the Experiment
EXP="ALPHA"
# mask checkpoint
MCHECK="/content/drive/MyDrive/HYDRA/models/splitnet.pth.tar"
# model type
M="splitnet"

python src/eval_alpha1.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${DATASET}/${M}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --model_type ${M}

DATASET="alpha1-L"

python src/eval_alpha1.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${DATASET}/${M}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --model_type ${M}