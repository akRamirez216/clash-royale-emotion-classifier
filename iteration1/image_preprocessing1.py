import kagglehub
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Download dataset
path = kagglehub.dataset_download("ananthu017/emotion-detection-fer")

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

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,), (0.229,))
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