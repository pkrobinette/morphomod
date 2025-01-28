#!/bin/bash

# dataset name
DATASET="alpha1-S"
# Name of the Experiment
EXP="STEPS"
# The dilation param
DILATE=3
# prompt to use for inpainting
PROMPT="Restore the natural texture."
# mask checkpoint
MCHECK="/content/drive/MyDrive/HYDRA/models/SLBR.pth.tar"
# refine checkpoint
RCHECK="/content/drive/MyDrive/HYDRA/train_refine_mask/CLWD_mask_refine/model/CLWD_mask_refine.pth.tar"
# model type
M="morphomod_slbr"
# fill type

NUMSTEPS=10
python src/eval_steps.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${NUMSTEPS}_P10_d${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt "${PROMPT}" \
    --num_steps ${NUMSTEPS}

NUMSTEPS=20
python src/eval_steps.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${NUMSTEPS}_P10_d${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt "${PROMPT}" \
    --num_steps ${NUMSTEPS}

NUMSTEPS=30
python src/eval_steps.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${NUMSTEPS}_P10_d${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt "${PROMPT}" \
    --num_steps ${NUMSTEPS}

NUMSTEPS=40
python src/eval_steps.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${NUMSTEPS}_P10_d${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt "${PROMPT}" \
    --num_steps ${NUMSTEPS}

NUMSTEPS=60
python src/eval_steps.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${NUMSTEPS}_P10_d${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt "${PROMPT}" \
    --num_steps ${NUMSTEPS}

NUMSTEPS=70
python src/eval_steps.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${NUMSTEPS}_P10_d${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt "${PROMPT}" \
    --num_steps ${NUMSTEPS}

NUMSTEPS=80
python src/eval_steps.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${NUMSTEPS}_P10_d${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt "${PROMPT}" \
    --num_steps ${NUMSTEPS}

NUMSTEPS=90
python src/eval_steps.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${NUMSTEPS}_P10_d${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt "${PROMPT}" \
    --num_steps ${NUMSTEPS}

NUMSTEPS=100
python src/eval_steps.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${NUMSTEPS}_P10_d${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt "${PROMPT}" \
    --num_steps ${NUMSTEPS}




