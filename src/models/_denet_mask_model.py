from argparse import Namespace
import os
import torch
import _src_denet.networks as nets

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class DENetMask():
    def __init__(self, mode="gray", checkpoint=None):
        """
        Args:
            mode: "gray", "h", or "l".
            checkpoint: the direct path to the checkpoint.
        """
        args = get_default_config()
        self.model = nets.__dict__['slbr'](args=args)
        self.device = DEVICE

        #
        # Load checkpoint
        #
        if not checkpoint: checkpoint = os.path.join(args.resume, f"denet_logo-{mode}.pth.tar")
        assert os.path.exists(checkpoint), f"Whoops, this {checkpoint} does not exist."
        self._load_checkpoint(checkpoint)

            

    def _load_checkpoint(self, checkpoint):
        """Load Checkpoint"""
        if not os.path.exists(checkpoint):
            raise Exception("=> no checkpoint found at '{}'".format(checkpoint))

        current_checkpoint = torch.load(checkpoint, map_location=torch.device(self.device), weights_only=True)
        try:
            self.model.load_state_dict(current_checkpoint['state_dict'], strict=True)
        except:
            new_state_dict = {key.replace("module.", ""): value for key, value in current_checkpoint['state_dict'].items()}
            self.model.load_state_dict(new_state_dict, strict=True)
        else:
            raise ValueError("Incorrect loading of model")

        print("=> loaded DENet checkpoint '{}'".format(checkpoint))


    def __call__(self, x):
        _, immask_all, _, _ = self.model(x)

        return immask_all[0]


def get_default_config():
    return Namespace(
        use_refine=True,
        k_skip_stage=3,
        mask_mode = 'res',
        k_center = 2,
        k_refine = 3,
        resume="/Users/probinette/Documents/PROJECTS/hydra_steg/models"
    )
