from argparse import Namespace
import os
import torch
import _src_wdnet.WDNet as wdnetm
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT = '/Users/probinette/Documents/PROJECTS/hydra_steg/models/wdnet_g.pkl'

class WDNetMask():
    def __init__(self, checkpoint=None):
        self.model = wdnetm.generator(3,3)
        self.device = DEVICE
        #
        # Load checkpoint
        #
        if not checkpoint: checkpoint = CHECKPOINT
        self._load_checkpoint(checkpoint)

    
    def _load_checkpoint(self, checkpoint):
        """Load Checkpoint"""
        if not os.path.exists(checkpoint):
            raise Exception("=> no checkpoint found at '{}'".format(checkpoint))

        current_checkpoint = torch.load(checkpoint, map_location=torch.device(self.device), weights_only=True)

        # ---------------- Load Model Weights --------------------------------------
        self.model.load_state_dict(current_checkpoint)
        print("=> loaded WDNet checkpoint '{}'".format(checkpoint))


    def __call__(self, x):
        _,mask,_,_,_ = self.model(x)

        return mask