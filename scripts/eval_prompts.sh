#!/bin/bash

# PROMPTS 
# P1: "Remove."
# P2: "Fill in the background."
# P3: "Erase the mark and restore the original."
# P4: "Blend into the surrounding area."
# P5: "Reconstruct the missing details."
# P6: "Remove the object and match the background."
# P7: "Fill in the gaps as if the mark was never there."
# P8: "Smooth out and complete the scene."
# P9: "Mend the area to look natural."
# P10: "Restore the natural texture."

# dataset name
DATASET="alpha1-S"
# Name of the Experiment
EXP="PROMPTS"
# The dilation param
DILATE=3
# prompt to use for inpainting
PROMPT="Remove."
# mask checkpoint
MCHECK="/content/drive/MyDrive/HYDRA/models/SLBR.pth.tar"
# refine checkpoint
RCHECK="/content/drive/MyDrive/HYDRA/train_refine_mask/CLWD_mask_refine/model/CLWD_mask_refine.pth.tar"
# model type
M="morphomod_slbr"

EXP_NAME="P1"

python src/eval_prompts.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${EXP_NAME}_${M}_dilate_${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt "${PROMPT}"


PROMPT="Fill in the background."
EXP_NAME="P2"

python src/eval_prompts.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${EXP_NAME}_${M}_dilate_${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt "${PROMPT}"

PROMPT="Erase the mark and restore the original."
EXP_NAME="P3"

python src/eval_prompts.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${EXP_NAME}_${M}_dilate_${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt "${PROMPT}"

PROMPT="Blend into the surrounding area."
EXP_NAME="P4"

python src/eval_prompts.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${EXP_NAME}_${M}_dilate_${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt "${PROMPT}"

PROMPT="Reconstruct the missing details."
EXP_NAME="P5"

python src/eval_prompts.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${EXP_NAME}_${M}_dilate_${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt "${PROMPT}"

PROMPT="Remove the object and match the background."
EXP_NAME="P6"

python src/eval_prompts.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${EXP_NAME}_${M}_dilate_${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt "${PROMPT}"

PROMPT="Fill in the gaps as if the mark was never there."
EXP_NAME="P7"

python src/eval_prompts.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${EXP_NAME}_${M}_dilate_${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt "${PROMPT}"

PROMPT="Smooth out and complete the scene."
EXP_NAME="P8"

python src/eval_prompts.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${EXP_NAME}_${M}_dilate_${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt "${PROMPT}"

PROMPT="Mend the area to look natural."
EXP_NAME="P9"

python src/eval_prompts.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${EXP_NAME}_${M}_dilate_${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt "${PROMPT}"

PROMPT="Restore the natural texture."
EXP_NAME="P10"

python src/eval_prompts.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${EXP_NAME}_${M}_dilate_${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt "${PROMPT}"
