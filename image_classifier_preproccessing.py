"""Project: Clash Royale Emotion Classifier
File: image_classifier_preprocessing.py
Description: This script downloads the FER emotion detection dataset from Kaggle,
preprocesses the images, and prepares them for training an image classifier.

Authors: Asher Kelly, Zai Yang
Date: June 2024
"""


import kagglehub
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Download latest version
path = kagglehub.dataset_download("ananthu017/emotion-detection-fer")



# Load and preprocess the dataset
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(size=48, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.ImageFolder(root=f"{path}/train", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{path}/test", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

   
    

