#!/bin/bash


# # data path
# DATAP="/Users/probinette/Documents/PROJECTS/hydra_steg/data/results/alpha_fill"
# # save path
# SAVEP="/Users/probinette/Documents/PROJECTS/hydra_steg/RESULTS/fill_exprs"

# python perception_stats/matlab_perception_eval_fill.py \
#     --data_path "${DATAP}" \
#     --save_path "${SAVEP}"


# data path
DATAP="/Users/probinette/Documents/PROJECTS/hydra_steg/data/results/CLWD_fill"
# save path
SAVEP="/Users/probinette/Documents/PROJECTS/hydra_steg/RESULTS/fill_exprs/CLWD"

python perception_stats/matlab_perception_eval_fill.py \
    --data_path "${DATAP}" \
    --save_path "${SAVEP}"
