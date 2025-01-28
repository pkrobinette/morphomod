
EXP="STEG"
DATASET="steg_disorient"

python steg_disorient/eval_steg_disorient.py \
    --expr_path "/content/drive/MyDrive/HYDRA/${EXP}/eval" \
    --data_path "/content/data/${DATASET}" \
    --class_checkpoint "/content/drive/MyDrive/HYDRA/STEG/models/steg_model.pth.tar" \
    --semseg_checkpoint "/content/drive/MyDrive/HYDRA/STEG/models/box_seg_unet.pth.tar" 
