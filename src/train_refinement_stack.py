"""
Training file to finetune slbr for semseg ;>
"""

from modules.module_refine_stack import RefineModuleStack
import pytorch_lightning as pl
import os
import torch
import pickle
import utils
import glob
import models
import datasets   
import argparse
import configs
import os.path as osp
import sys
from datetime import datetime

# get device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"
print(DEVICE)


def get_args():
    """Get args for experiment"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--expr_path", type=str, help="Where to save results.")
    parser.add_argument("--save_name", type=str, default="model", help="The model save name.")
    parser.add_argument("--test", action="store_true", help="just testing model")
    parser.add_argument(
        "-d", 
        "--dataset", 
        type=str, 
        default="refine_10kgray", 
        help="The dataset to use. ['refine_10kgray']"
    )
    parser.add_argument("--data_path", type=str, help="path of the dataset")
    parser.add_argument(
        "-c", 
        "--checkpoint", 
        type=str, 
        default=None,
        help="path to the checkpoint."
    )
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")

    args = parser.parse_args()

    return args
    

def get_dataset(is_train: str, args: argparse.Namespace):
    """Load dataset.

    Args:
        dataset: key for the dataset
        is_train: "test" or "train"
    Returns:
        DataLoader
    """
    if args.dataset == "refine_10kgray":
        return datasets.load_refine10kgray(is_train=is_train, batch_size=args.batch_size, path=args.data_path)

    raise ValueError("Dataset does not exist.")
    

def main():
    """ The main FT function """
    args = get_args()  
    utils.pprint_args(args)
    args.expr_path += "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(args.expr_path, exist_ok=True)
    #
    # Init model
    #
    print("Initializing models ...")
    refine_model = models.UNetRefineStack()
    if args.checkpoint is not None:
        refine_model = utils.load_checkpoint(refine_model, args.checkpoint)
    model = RefineModuleStack(
        model=refine_model,      # model
        lr=0.001,        # learning rate
        num_images_save=10, # number of images to save during testing
        save_path=osp.join(args.expr_path, "_images")
    )
    #
    # If training and test
    #
    if not args.test:
        #
        # Dataset
        #
        train_loader = get_dataset("train", args)
        #
        # Set up trainer
        #
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            devices=1,
            accelerator="gpu" if DEVICE == "cuda" else DEVICE,
        )
        #
        # Train
        #
        print("Training ...")
        trainer.fit(model=model, train_dataloaders=train_loader)
        #
        # Save model
        #
        os.makedirs(osp.join(args.expr_path, "model"), exist_ok=True)
        print("Model saved to: ", os.path.join(args.expr_path, "model", args.save_name + ".pth.tar"))
        torch.save({'state_dict': model.model.state_dict()}, os.path.join(args.expr_path, "model", args.save_name + ".pth.tar"))
    #
    # Test dataset
    #
    test_loader = get_dataset("test", args)
    #
    # Test
    #
    model.model.eval()
    trainer = pl.Trainer(
        max_epochs=1,
        devices=1,
        accelerator="gpu" if DEVICE == "cuda" else DEVICE,
    )
    
    metrics = trainer.test(model=model, dataloaders=test_loader)

    with open(os.path.join(args.expr_path, f'metrics.pkl'), 'wb') as file:
        pickle.dump(metrics[0], file)
    

if __name__ == "__main__":
    main()
