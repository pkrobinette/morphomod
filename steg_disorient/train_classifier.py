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
    model = CNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(args.epochs):
        model.train()  # set to training mode
        total_loss = 0.0
        correct = 0
        total = 0
    
        for batch in tqdm.tqdm(train_loader):
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)

    
            optimizer.zero_grad()
            outputs = model(imgs)       
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
    #
    # tes
    #
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader):
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)
    
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * imgs.size(0)
    
            _, preds = outputs.max(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    avg_test_loss = test_loss / total
    test_accuracy = 100.0 * correct / total
    print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

    #
    # save
    #
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    args = get_args()
    args.data_path = "/content/data/steg_disorient"
    args.save_path = "/content/drive/MyDrive/HYDRA/STEG/models/steg_model.pth.tar"

    main(args)
    
    