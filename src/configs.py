import argparse


def get_default_slbr_config():
    return argparse.Namespace(
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


def get_default_denet_config():
    return argparse.Namespace(
        use_refine=True,
        k_skip_stage=3,
        mask_mode = 'res',
        k_center = 2,
        k_refine = 3,
        resume="/Users/probinette/Documents/PROJECTS/hydra_steg/models"
    )