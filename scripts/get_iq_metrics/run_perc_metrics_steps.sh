#!/bin/bash


# data path
DATAP="/Users/probinette/Documents/PROJECTS/hydra_steg/data/results/alpha_steps"
# save path
SAVEP="/Users/probinette/Documents/PROJECTS/hydra_steg/RESULTS/steps_exprs"

python perception_stats/matlab_perception_eval_ft.py \
    --data_path "${DATAP}" \
    --save_path "${SAVEP}"
