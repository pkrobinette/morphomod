#!/bin/bash

# dataset name
DATASET="alpha1-S"
# Name of the Experiment
EXP="MODELS_EVAL"
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
INPAINTMOD="SDXL"


python src/eval_models.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${DATASET}/${INPAINTMOD}_P10_d${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt "${PROMPT}" \
    --inpaint_mod "${INPAINTMOD}"
    

DATASET="CLWD"

python src/eval_models.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/${DATASET}/${INPAINTMOD}_P10_d${DILATE}" \
    --data_path "/content/data/${DATASET}" \
    --dataset "${DATASET}" \
    --checkpoint ${MCHECK} \
    --refine_checkpoint ${RCHECK} \
    --model_type ${M} \
    --dilate ${DILATE} \
    --prompt "${PROMPT}" \
    --inpaint_mod "${INPAINTMOD}"





