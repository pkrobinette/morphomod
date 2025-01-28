from argparse import Namespace
import os
import torch
import _src_slbr.networks as nets

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SLBRMask():
    def __init__(self, checkpoint=None):
        args = get_default_config()
        self.model = nets.__dict__['slbr_prime_mask'](args=args)
        self.device = DEVICE

        #
        # Load checkpoint
        #
        if not checkpoint: checkpoint = args.resume
        self._load_checkpoint(checkpoint)
            

    def _load_checkpoint(self, checkpoint):
        """Load Checkpoint"""
        if not os.path.exists(checkpoint):
            raise Exception("=> no checkpoint found at '{}'".format(checkpoint))

        current_checkpoint = torch.load(checkpoint, map_location=torch.device(self.device), weights_only=True)
        if isinstance(current_checkpoint['state_dict'], torch.nn.DataParallel):
            current_checkpoint['state_dict'] = current_checkpoint['state_dict'].module

        # ---------------- Load Model Weights --------------------------------------
        self.model.load_state_dict(current_checkpoint['state_dict'], strict=True)
        print("=> loaded SLBR checkpoint '{}'".format(checkpoint))


    def __call__(self, x):
        return self.model(x)

        


def get_default_config():
    return Namespace(
        input_size=256,
        crop_size=256,
        dataset_dir="../data/CLWD",
        preprocess="resize",
        no_flip=True,
        nets="slbr",
        models="slbr",   resume="/Users/probinette/Documents/PROJECTS/hydra_steg/models/SLBR.pth.tar",
        bg_mode="res_mask",
        mask_mode="res",
        sim_metric="cos",
        k_center=2,
        project_mode="simple",
        use_refine=True,
        k_refine=3,
        k_skip_stage=3,
        name="slbr_v1",
        checkpoint="checkpoint",
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0,
        evaluate=True,
        hl=False,
        lambda_style=0,
        lambda_content=0,
        lambda_iou=0,
        lambda_mask=1,
        lambda_primary=0.01,
        start_epoch=0,
        schedule=[5, 10],
        gamma=0.1,
        gan_norm=False
    )

