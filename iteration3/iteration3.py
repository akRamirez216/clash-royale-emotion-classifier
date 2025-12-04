import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from image_classifier_preproccessing import get_dataloaders
import torchvision.models as models
import os
import json

# ======================
# SETUP
# ======================
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("\nTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("# of GPUs:", torch.cuda.device_count())

# -----------------------
# CONFIG
# -----------------------
NUM_CLASSES = 7
BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-4                        # lower LR for transfer learning
WEIGHT_DECAY = 1e-4
PATIENCE = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = torch.cuda.amp.GradScaler()

# -----------------------
# LOAD DATA
# -----------------------
train_loader, val_loader, class_names = get_dataloaders(
    batch_size=BATCH_SIZE,
    num_workers=4
)

print("\nClasses:", class_names)
print("Using device:", DEVICE)
print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))


# -----------------------
# RESNET18 TRANSFER MODEL
# -----------------------
def get_resnet18_transfer(num_classes):
    
    model = models.resnet18(weights="IMAGENET1K_V1")

    # Change first conv layer to accept 1 channel instead of 3
    model.conv1 = nn.Conv2d(
        1, 64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    # Change classifier head
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )

    return model


model = get_resnet18_transfer(NUM_CLASSES).to(DEVICE)

# -----------------------
# FREEZE MOST LAYERS
# -----------------------
# Only train final layer + batchnorm layers first (faster & better)
for name, param in model.named_parameters():
    if "fc" not in name and "bn" not in name:
        param.requires_grad = False


# -----------------------
# LOSS / OPTIMIZER / SCHEDULER
# -----------------------
criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)


# -----------------------
# TRAIN / VALIDATE
# -----------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100 * correct / total


def validate_one_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100 * correct / total, all_labels, all_preds


# -----------------------
# MAIN
# -----------------------
def main():

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    final_labels = None
    final_preds = None

    best_val = 0
    patience_count = 0

    for epoch in range(EPOCHS):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion
        )

        val_loss, val_acc, labels, preds = validate_one_epoch(
            model, val_loader, criterion
        )

        scheduler.step(val_acc)

        final_labels = labels
        final_preds = preds

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # ------------------ UNFREEZE AFTER 10 EPOCHS ------------------
        if epoch == 10:
            print("\n🔓 Unfreezing all layers for fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True

            optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f" Val  Loss: {val_loss:.4f} | Val  Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val:
            best_val = val_acc
            patience_count = 0

            torch.save(model.state_dict(), f"{RESULTS_DIR}/best_transfer_resnet18.pth")

        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print("\n✅ Early stopping triggered")
                break


    # -----------------------
    # PLOTS
    # -----------------------
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


    # -----------------------
    # CONFUSION MATRIX
    # -----------------------
    cm = confusion_matrix(
        final_labels,
        final_preds,
        labels=list(range(NUM_CLASSES))
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


    # -----------------------
    # SAVE FINAL RESULTS
    # -----------------------
    results = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "best_val_acc": best_val,
        "classes": class_names,
        "confusion_matrix": cm.tolist()
    }

    with open(f"{RESULTS_DIR}/training_results_transfer.json", "w") as f:
        json.dump(results, f, indent=4)

    with open(f"{RESULTS_DIR}/summary_transfer.txt", "w") as f:
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LR}\n")
        f.write(f"Best Val Accuracy: {best_val:.2f}%\n")

    print("\n✅ Training Complete")
    print(f"✅ Best model saved: {RESULTS_DIR}/best_transfer_resnet18.pth")


if __name__ == "__main__":
    main()