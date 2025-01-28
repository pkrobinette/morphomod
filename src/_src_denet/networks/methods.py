from _src_denet.utils.model_init import *
from _src_denet.networks.resunet import SLBR, SLBRMask


# our method
def slbr(**kwargs):
    return SLBR(args=kwargs['args'], shared_depth=1, blocks=3, long_skip=True)

def slbr_mask(**kwargs):
    return SLBRMask(args=kwargs['args'], shared_depth=1, blocks=3, long_skip=True)



