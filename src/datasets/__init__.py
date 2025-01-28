# from .COCO import COCO
# from .BIH import BIH
from .clwd_dataset import CLWDDataset
from .lvw_dataset import LVWDataset
from .clwd_logo27_dataset import CLWD_LOGO27Dataset
from .COCO import COCO
from .BIH import BIH
from .refine_10k_dataset import Refine10kDataset
from .mask_refine_dataset import MaskRefineDataset
import importlib
import torch.utils.data
from datasets.base_dataset import BaseDataset

from ._load_clwd import load_clwd as load_clwd_dataset
from ._load_10k import load_10k
from ._load_clwd_logo27 import load_clwd_logo27_dataset

from ._load_refine10k import load_refine10k_dataset
from ._load_mask_refine import load_mask_refine_dataset

from .dilate_dataset import DilateDataset
from .alpha1_dataset import AlphaDataset
from .clwd_dataset_np import CLWDNPDataset
from .alpha1_dataset_np import AlphaNPDataset

from torch.utils.data import DataLoader
import os
from typing import Union


__all__ = ('CLWDDataset', 'LVWDataset', 'COCO','BIH', 'Refine10kDataset', 'MaskRefineDataset', 'CLWD_LOGO27Dataset')
 

def load_clwd(is_train: str = "test", batch_size: int = 8, shuffle=True, path=None):
    return load_clwd_dataset(is_train=is_train, batch_size=batch_size, shuffle=shuffle, path=path)

def load_10kgray(is_train: str = "test", batch_size: int = 8, shuffle=True, path=None):
    return load_10k(is_train=is_train, mode="gray", batch_size=batch_size, shuffle=shuffle, path=path)

def load_10kmid(is_train: str = "test", batch_size: int = 8, shuffle=True, path=None):
    return load_10k(is_train=is_train, mode="mid", batch_size=batch_size, shuffle=shuffle, path=path)

def load_10khigh(is_train: str = "test", batch_size: int = 8, shuffle=True, path=None):
    return load_10k(is_train=is_train, mode="high", batch_size=batch_size, shuffle=shuffle, path=path)

def load_clwd_logo27(is_train: str = 'test', batch_size: int = 8, shuffle=True, path=None):
    return load_clwd_logo27_dataset(is_train=is_train, batch_size=batch_size, shuffle=shuffle, path=path)

def load_refine10kgray(is_train: str = "test", batch_size: int = 8, shuffle=True, path=None):
    return load_refine10k_dataset(is_train=is_train, mode="gray", batch_size=batch_size, shuffle=shuffle, path=path)

def load_mask_refine(is_train: str = "test", batch_size: int = 8, shuffle=True, path=None):
    return load_mask_refine_dataset(is_train=is_train, batch_size=batch_size, shuffle=shuffle, path=path)

def load_dilate_dataset(
    dilate_val: int, 
    path: str, 
    batch_size: int=8, 
    shuffle: bool = True, 
    num_workers: int = 2, 
    pin_memory: bool = True
    ):
    dataset = DilateDataset(dilate_val, path)

    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

def load_alpha1_dataset(
    is_train: str,
    path: str, 
    batch_size: int=8, 
    shuffle: bool = True, 
    num_workers: int = 2, 
    pin_memory: bool = True
    ):
    dataset = AlphaDataset(is_train, path)

    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

def load_clwdnp_dataset(
    is_train: str,
    path: str, 
    num_samples: Union[int, None] = None,
    batch_size: int=8, 
    shuffle: bool = True, 
    num_workers: int = 2, 
    pin_memory: bool = True
    ):
    dataset = CLWDNPDataset(is_train, path, num_samples)

    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

def load_alphanp_dataset(
    is_train: str,
    path: str, 
    num_samples: Union[int, None] = None,
    batch_size: int=8, 
    shuffle: bool = True, 
    num_workers: int = 2, 
    pin_memory: bool = True
    ):
    dataset = AlphaNPDataset(is_train, path, num_samples)

    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
