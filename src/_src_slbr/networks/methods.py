

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import math

from _src_slbr.utils.model_init import *
from _src_slbr.networks.resunet import SLBR, SLBRPrimeMask


# our method
def slbr(**kwargs):
    return SLBR(args=kwargs['args'], shared_depth=1, blocks=3, long_skip=True)

def slbr_prime_mask(**kwargs):
    return SLBRPrimeMask(args=kwargs['args'], shared_depth=1, blocks=3, long_skip=True)




