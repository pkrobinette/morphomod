#!/bin/bash

# dataset name
DATASET="10kgray"
# Name of the Experiment
EXP="DILATE"
# The dilation param
DILATE=0
# prompt to use for inpainting
PROMPT="Remove."
# mask checkpoint
MCHECK="/content/drive/MyDrive/HYDRA/models/denet_logo-gray.pth.tar"
# refine checkpoint
RCHECK="/content/drive/MyDrive/HYDRA/train_refine_mask/${DATASET}_mask_refine/model/${DATASET}_mask_refine.pth.tar"
# model type
M="denet-g"

python src/eval_morphomod.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${DATASET}/dilate_${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --mask_checkpoint ${MCHECK} \
    --mask_refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --save \
    --dilate ${DILATE} \
    --prompt ${PROMPT} 


# The dilation param
DILATE=1

python src/eval_morphomod.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${DATASET}/dilate_${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --mask_checkpoint ${MCHECK} \
    --mask_refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --save \
    --dilate ${DILATE} \
    --prompt ${PROMPT} 


# The dilation param
DILATE=3

python src/eval_morphomod.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${DATASET}/dilate_${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --mask_checkpoint ${MCHECK} \
    --mask_refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --save \
    --dilate ${DILATE} \
    --prompt ${PROMPT} 

# The dilation param
DILATE=5

python src/eval_morphomod.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${DATASET}/dilate_${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --mask_checkpoint ${MCHECK} \
    --mask_refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --save \
    --dilate ${DILATE} \
    --prompt ${PROMPT} 

# The dilation param
DILATE=10

python src/eval_morphomod.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${DATASET}/dilate_${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --mask_checkpoint ${MCHECK} \
    --mask_refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --save \
    --dilate ${DILATE} \
    --prompt ${PROMPT} 