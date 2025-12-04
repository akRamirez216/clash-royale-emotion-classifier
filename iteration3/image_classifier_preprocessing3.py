"""
Project: Clash Royale Emotion Classifier
File: image_classifier_preprocessing.py
Description: Downloads the FER emotion dataset and prepares PyTorch DataLoaders.

Authors: Asher Kelly, Zai Yang
Date: June 2024
"""

import kagglehub
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Download dataset
path = kagglehub.dataset_download("ananthu017/emotion-detection-fer")

# -----------------------
# TRAIN TRANSFORMS (with augmentation)
# -----------------------
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomAffine(
            degrees=25,  # Max rotation angle (you can increase this, e.g., 25)
            translate=(0.1, 0.1) # Max horizontal and vertical translation (10%)
        ),

    transforms.RandomResizedCrop(size=48, scale=(0.7, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -----------------------
# TEST / VAL TRANSFORMS (NO AUGMENTATION)
# -----------------------
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def get_dataloaders(batch_size=64, num_workers=4):
    train_dataset = datasets.ImageFolder(
        root=f"{path}/train",
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        root=f"{path}/test",
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader, train_dataset.classes
