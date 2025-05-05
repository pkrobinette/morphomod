from ._slbr_mask_model import SLBRMask
from ._splitnet_mask_model import SplitNetMask
from ._wdnet_mask_model import WDNetMask
from ._denet_mask_model import DENetMask

from _src_denet import networks as denet_nets
from _src_slbr import networks as slbr_nets
from _src_splitnet import models as splitnet_nets
from _src_wdnet.WDNet import generator, generator_mask

# from .unet_refine import UNetRefine
from .unet_refine_stack import UNetRefineStack
from .unet_refine_semseg import UNetRefineSemseg
# from .deeplab_refine_semseg import DeepLabRefineSemseg
from .morphomod import MorphoModel, dilate as Dilate


def load_slbr(checkpoint: str = None):
    return SLBRMask(checkpoint=checkpoint)

def load_splitnet(checkpoint: str = None):
    return SplitNetMask(checkpoint=checkpoint)

def load_wdnet(checkpoint: str = None):
    return WDNetMask(checkpoint=checkpoint)

def load_denet(mode: str, checkpoint: str = None):
    return DENetMask(mode=mode, checkpoint=checkpoint)



