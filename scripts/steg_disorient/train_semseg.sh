EXP="STEG"
DATASET="steg_disorient"

python steg_disorient/train_semseg.py \
    --save_path "/content/drive/MyDrive/HYDRA/${EXP}/eval" \
    --data_path "/content/data/${DATASET}" \
    --epochs 10
