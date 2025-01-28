from cnn import CNN
from steg_dataset import StegDisorientDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import os
import tqdm
import argparse
import segmentation_models_pytorch as smp


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default=None, help="location of dataset.")
    parser.add_argument("--save_path", type=str, default=None, help="location to save model.")
    parser.add_argument("--epochs", type=int, default=5, help="Num of train epochs")

    args = parser.parse_args()

    return args

def main(args):
    #
    # get datasets
    #
    train_dataset = StegDisorientDataset(is_train='train', root_dir=args.data_path)
    test_dataset  = StegDisorientDataset(is_train='test',  root_dir=args.data_path)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=True, num_workers=2)
    #
    # Train
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = smp.Unet(
        encoder_name="mobilenet_v2",   
        encoder_weights="imagenet",
        in_channels=3,
        classes=1, 
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()     
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # -------------------
    # Training Loop
    # -------------------
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm.tqdm(train_loader):
            imgs = batch['image'].to(device)   # (B, 3, H, W)
            masks = batch['mask'].to(device) # (B, 1, H, W)
            optimizer.zero_grad()
            logits = model(imgs)     # (B, 1, H, W)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_loss:.4f}")

    # -------------------
    # Testing (Evaluation)
    # -------------------
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
              imgs = batch['image'].to(device)
              masks = batch['mask'].to(device)
              logits = model(imgs)   # shape: (B, 1, H, W)
              loss = criterion(logits, masks)
              test_loss += loss.item() * imgs.size(0)
        test_loss /= len(test_dataset)
        print(f"Final Test Loss: {test_loss:.4f}")

    # -------------------
    # Save Model
    # -------------------
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to '{args.save_path}'")

    
if __name__ == "__main__":
    args = get_args()
    args.data_path = "/content/data/steg_disorient"
    args.save_path = "/content/drive/MyDrive/HYDRA/STEG/models/box_seg_unet.pth.tar"

    main(args)
    
    