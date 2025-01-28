from argparse import Namespace
import os
import torch
import _src_splitnet.models as archs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SplitNetMask():
    def __init__(self, checkpoint=None):
        args = get_default_config()
        self.model = archs.__dict__[args.arch]()
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
        self.model.load_state_dict(current_checkpoint['state_dict'])
        print("=> loaded SplitNet checkpoint '{}'".format(checkpoint))


    def __call__(self, x):
        _, immask_all, _ = self.model(x)

        return immask_all


def get_default_config():
    return Namespace(
        arch="vvv4n",
        resume="/Users/probinette/Documents/PROJECTS/hydra_steg/models/splitnet.pth.tar",
    )