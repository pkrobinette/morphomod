#!/bin/bash

# dataset name
DATASET="alpha1-S"
# Name of the Experiment
EXP="FILL"
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
FILL="background"

python src/eval_fill.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${FILL}_P10_dilate_${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt "${PROMPT}" \
    --fill "${FILL}"

FILL="black"
python src/eval_fill.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${FILL}_P10_dilate_${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt "${PROMPT}" \
    --fill "${FILL}"

FILL="white"
python src/eval_fill.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${FILL}_P10_dilate_${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt "${PROMPT}" \
    --fill "${FILL}"

FILL="gray"
python src/eval_fill.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${FILL}_P10_dilate_${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt "${PROMPT}" \
    --fill "${FILL}"

