from argparse import Namespace
import datasets
from torch.utils.data import DataLoader
import os
from typing import Optional


def get_default_config():
    return Namespace(
        input_size=256,
        crop_size=256,
        dataset_dir=(
            "/Users/probinette/Documents/PROJECTS/hydra_steg/data/CLWD"
        ),
        preprocess="resize",
        no_flip=True,
    )
    

def load_refine10k_dataset(
    mode: str,
    is_train: str = "test",
    batch_size: int = 8, 
    shuffle: bool = True, 
    num_workers: int = 2, 
    pin_memory: bool = True,
    path: Optional[str] = None,
):
    """
    Load the REFINE 10kgray Watermark Dataset.
    
    Args:
        mode: "gray, mid (l), or high (h)"
        batch_size: The size of the batch.
        shuffle: If True, the data will be shuffled.
        num_workers: The number of worker threads to use for loading data.
        pin_memory: If True, data will be loaded into pinned memory for faster GPU transfer.
        path: path of the dataset.
    """
    args = get_default_config()
    if path:
        args.dataset_dir = path
    assert os.path.exists(args.dataset_dir), f"{args.dataset_dir} does not exist :)"
    dataset = datasets.Refine10kDataset(is_train, args)

    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    