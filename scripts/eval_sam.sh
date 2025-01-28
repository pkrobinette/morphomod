python sam/eval_sam_auto_mask.py --dataset "CLWD"
python sam/eval_sam_auto_mask.py --dataset "Alpha1-S"

python sam/eval_sam_gdino_bbox.py --dataset "CLWD"
python sam/eval_sam_gdino_bbox.py --dataset "Alpha1-S"

python sam/eval_sam_gt_bbox.py --dataset "CLWD"
python sam/eval_sam_gt_bbox.py --dataset "Alpha1-S"

python sam/eval_sam_gt_points.py --dataset "CLWD"
python sam/eval_sam_gt_points.py --dataset "Alpha1-S"