import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os
import numpy as np
#from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
from tqdm import tqdm

transform = transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.RandomVerticalFlip(p=0.5),  # Flips vertically with a probability of 50%
                            transforms.RandomRotation(degrees=30),  # Rotates image randomly within the range [-30, 30] degrees
                            transforms.RandomHorizontalFlip(p=0.5),  # Flips horizontally with a probability of 50%
                            transforms.ToTensor(),
                              ])

def collate_fn(batch):
    images, labels = zip(*batch)
    return {
        'pixel_values': torch.stack(images),
        'labels': torch.tensor(labels)
    }


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Since there are no classes, we can return a dummy label (e.g., 0)
        return image, 0

def create_dataset(img_dir='/home1/eswara_e/VAE/img_align_celeba',batch_size=64):
    
    dataset = CustomImageDataset(root_dir=img_dir, transform=transform)
    #test_dataset = CustomImageDataset(root_dir='stanford_cars/cars_test', transform=transform)

    train_size = int(0.7 * len(dataset))   # 70% for training
    test_size = int(0.20 * len(dataset))     # 20% for test
    val_size = len(dataset) - train_size - test_size  # Remaining 10% for val


    train_dataset,test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers = 4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    # print(len(dataset))
    # print(len(train_dataset))
    # print(len(test_dataset))
    # print(len(val_dataset))
    return train_loader,test_loader,val_loader
