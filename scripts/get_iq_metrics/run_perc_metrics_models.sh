#!/bin/bash


# # data path
# DATAP="/Users/probinette/Documents/PROJECTS/hydra_steg/data/results/alpha1-S_models"
# # save path
# SAVEP="/Users/probinette/Documents/PROJECTS/hydra_steg/RESULTS/MODELS_EVAL/alpha1-S_models"

# MODEL="LaMa"

# python perception_stats/matlab_perception_eval_models.py \
#     --data_path "${DATAP}" \
#     --save_path "${SAVEP}" \
#     --model "${MODEL}"

# MODEL="SDXL"
# python perception_stats/matlab_perception_eval_models.py \
#     --data_path "${DATAP}" \
#     --save_path "${SAVEP}" \
#     --model "${MODEL}"


# data path
DATAP="/Users/probinette/Documents/PROJECTS/hydra_steg/data/results/CLWD_models"
# save path
SAVEP="/Users/probinette/Documents/PROJECTS/hydra_steg/RESULTS/MODELS_EVAL/CLWD_models"

MODEL="LaMa"

python perception_stats/matlab_perception_eval_models.py \
    --data_path "${DATAP}" \
    --save_path "${SAVEP}" \
    --model "${MODEL}"

MODEL="SDXL"
python perception_stats/matlab_perception_eval_models.py \
    --data_path "${DATAP}" \
    --save_path "${SAVEP}" \
    --model "${MODEL}"