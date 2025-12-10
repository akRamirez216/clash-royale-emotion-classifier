import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from image_classifier_preproccessing import get_dataloaders
import os
import json

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("# of GPUs:", torch.cuda.device_count())


# Config
NUM_CLASSES = 7
BATCH_SIZE = 64
EPOCHS = 25
LR = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
train_loader, val_loader, class_names = get_dataloaders(
    batch_size=BATCH_SIZE,
    num_workers=2
)

print("Classes:", class_names)
print("Using device:", DEVICE)
print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))


# Basic Residual Block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# Resnet Model
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=7):
        super(ResNet, self).__init__()
        self.in_channels = 32

        self.conv1 = nn.Conv2d(
            1, 32,
            kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(32)

        self.layer1 = self._make_layer(block, 32,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64,  num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


def ResNet18_emotion(num_classes=7):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


# Train / Validate Functions
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100 * correct / total


def validate_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100 * correct / total, all_labels, all_preds


# Main
def main():

    model = ResNet18_emotion(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    final_labels = None
    final_preds = None

    best_val_acc = 0.0
    best_model_path = f"{RESULTS_DIR}/best_emotion_resnet18.pth"

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion
        )

        val_loss, val_acc, labels, preds = validate_one_epoch(
            model, val_loader, criterion
        )

        # Save final epoch predictions for confusion matrix
        final_labels = labels
        final_preds = preds

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f" Val  Loss: {val_loss:.4f} | Val  Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"🔥 Best model updated (epoch {epoch+1}) | Val Acc: {val_acc:.2f}%")


    # Plots
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/loss_curve.png")
    plt.show()

    plt.figure()
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/accuracy_curve.png")
    plt.show()


    # Confusion Matrix
    cm = confusion_matrix(
        final_labels,
        final_preds,
        labels=list(range(NUM_CLASSES))   # force full 7x7
    )

    plt.figure(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )
    disp.plot(xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png")
    plt.show()


    # Save Results
    results = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "classes": class_names,
        "confusion_matrix": cm.tolist()
    }

    with open(f"{RESULTS_DIR}/training_results.json", "w") as f:
        json.dump(results, f, indent=4)


    with open(f"{RESULTS_DIR}/summary.txt", "w") as f:
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LR}\n")
        f.write(f"Final Train Accuracy: {train_accs[-1]:.2f}%\n")
        f.write(f"Final Val Accuracy: {val_accs[-1]:.2f}%\n")


    # Save Model
    torch.save(model.state_dict(), f"{RESULTS_DIR}/emotion_resnet18_final.pth")


    print(f"\n✅ Model saved as {RESULTS_DIR}/emotion_resnet18_final.pth")
    print(f"✅ Confusion Matrix saved as {RESULTS_DIR}/confusion_matrix.png")
    print(f"✅ Results saved as training_results.json and summary.txt")


if __name__ == "__main__":
    main()