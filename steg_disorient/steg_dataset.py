import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import os


class StegDisorientDataset(Dataset):
    """
    Loads images and masks from all four directions: up, down, left, right.
    Returns (image, mask, direction).
    """
    def __init__(self, is_train: str, root_dir):
        """
        Args:
            root_dir (str): Path to the 'test' folder.
            directions (list of str, optional): A subset of ['up', 'down', 'left', 'right'].
        """
        self.root_dir = os.path.join(root_dir, is_train)
        self.transform = T.ToTensor()

        self.class_map = {
            "up": 0,
            "right": 1,
            "down": 2,
            "left": 3
        }

        self.samples = []
        # For each direction, collect paired image/mask paths
        for d in self.class_map.keys():
            img_dir = os.path.join(self.root_dir, "image", d)
            mask_dir = os.path.join(self.root_dir, "mask", d)

            # Gather valid images
            for fname in sorted(os.listdir(img_dir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(img_dir, fname)
                    mask_path = os.path.join(mask_dir, fname)
                    self.samples.append((img_path, mask_path, d))



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, direction = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = self.transform(img)
        mask = self.transform(mask)
        mask = (mask > 0.5).float()

        label = torch.tensor(self.class_map[direction], dtype=torch.long)

        return {
            "image": img,
            "mask": mask,
            "label": label,
            "id": os.path.basename(img_path).replace(".jpg", "")
        }