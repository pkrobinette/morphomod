#!/bin/bash


# # data path
# DATAP="/Users/probinette/Documents/PROJECTS/hydra_steg/data/results/alpha1-S"
# # save path
# SAVEP="/Users/probinette/Documents/PROJECTS/hydra_steg/RESULTS/prompt_exprs"

# python src/matlab_perception_eval_ft.py \
#     --data_path "${DATAP}" \
#     --save_path "${SAVEP}"


# data path
DATAP="/Users/probinette/Documents/PROJECTS/hydra_steg/data/results/clwd_prompts"
# save path
SAVEP="/Users/probinette/Documents/PROJECTS/hydra_steg/RESULTS/prompt_exprs/CLWD"

python perception_stats/matlab_perception_eval_ft.py \
    --data_path "${DATAP}" \
    --save_path "${SAVEP}"