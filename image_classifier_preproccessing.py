"""
This module handles the preprocessing and loading of the image dataset for emotion detection.
It includes functions to download the dataset, apply necessary transformations, and create data loaders for training and testing.

Authors: Asher Kelly, Zai Yang
Date: December 13, 2025
"""



import kagglehub
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Download dataset
path = kagglehub.dataset_download("ananthu017/emotion-detection-fer")

# Define transformations for training and testing datasets
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomAffine(
    degrees=15,
    translate=(0.05, 0.05),
    scale=(0.9, 1.05)
    ),

    transforms.ToTensor(),
    transforms.Normalize((0.485,), (0.229,))
])

# Define transformations for testing dataset
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,), (0.229,))
])


def get_dataloaders(batch_size=64, num_workers=4):
    """
    Create and return training and testing data loaders.
    
    :param batch_size: Specifies the number of samples per batch to load.
    :param num_workers: Specifies the number of subprocesses to use for data loading.
    :return: A tuple containing the training data loader, testing data loader, and class names.
    """

    # Load datasets
    train_dataset = datasets.ImageFolder(
        root=f"{path}/train",
        transform=train_transform
    )

    # Load test dataset
    test_dataset = datasets.ImageFolder(
        root=f"{path}/test",
        transform=test_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader, train_dataset.classes