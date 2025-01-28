#!/bin/bash

# dataset name
DATASET="alpha1-S"
# Name of the Experiment
EXP="ALPHA"
# The dilation param
DILATE=0
# prompt to use for inpainting
PROMPT="Remove."
# mask checkpoint
MCHECK="/content/drive/MyDrive/HYDRA/models/SLBR.pth.tar"
# refine checkpoint
RCHECK="/content/drive/MyDrive/HYDRA/train_refine_mask/CLWD_mask_refine/model/CLWD_mask_refine.pth.tar"
# model type
M="morphomod_slbr"

python src/eval_alpha1.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${DATASET}/${M}-dilate_${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt ${PROMPT} 


# The dilation param
DILATE=1

python src/eval_alpha1.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${DATASET}/${M}-dilate_${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt ${PROMPT} 


# The dilation param
DILATE=3

python src/eval_alpha1.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${DATASET}/${M}-dilate_${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt ${PROMPT} 

# The dilation param
DILATE=5

python src/eval_alpha1.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${DATASET}/${M}-dilate_${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt ${PROMPT} 

# The dilation param
DILATE=10

python src/eval_alpha1.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${DATASET}/${M}-dilate_${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt ${PROMPT} 