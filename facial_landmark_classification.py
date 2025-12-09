"""
Emotion recognition with Mini-Xception using the Kaggle FER dataset.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.layers import (
    Input, SeparableConv2D, BatchNormalization,
    ReLU, GlobalAveragePooling2D, Dense
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# CONSTANTS
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
DATASET_PATH = kagglehub.dataset_download("ananthu017/emotion-detection-fer")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_emotion_model.keras")
WEIGHTS_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_weights.h5")

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = len(EMOTIONS)


# MODEL: Mini-Xception
def build_mini_xception(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = SeparableConv2D(8, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = SeparableConv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = SeparableConv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)


# DATA LOADER
def load_split_data(root_path):
    def load_folder(folder_path):
        data, labels = [], []
        for idx, emotion in enumerate(EMOTIONS):
            emotion_dir = os.path.join(folder_path, emotion.lower())
            if not os.path.isdir(emotion_dir):
                print(f"[WARN] Missing folder: {emotion_dir}")
                continue

            for fname in os.listdir(emotion_dir):
                fpath = os.path.join(emotion_dir, fname)

                img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                img = cv2.resize(img, IMG_SIZE)
                img = img.astype("float32") / 255.0

                data.append(img)
                labels.append(idx)

        return np.array(data), np.array(labels)

    print("Loading training set...")
    X_train, y_train = load_folder(os.path.join(root_path, "train"))

    print("Loading test set...")
    X_test, y_test = load_folder(os.path.join(root_path, "test"))

    return X_train, y_train, X_test, y_test


# TRAINING
def train_and_evaluate(model, X_train, y_train, X_test, y_test):

    # Save the best .keras model
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, early_stop]
    )

    # Save training plots
    # Accuracy plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.legend(); plt.grid(); plt.title("Accuracy")
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_plot.png"))
    plt.close()

    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.legend(); plt.grid(); plt.title("Loss")
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_plot.png"))
    plt.close()

    # Save weights separately
    model.save_weights(WEIGHTS_SAVE_PATH)

    return history


def save_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=labels, yticklabels=labels,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()


def write_summary(train_acc, val_acc):
    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Device: {tf.config.list_physical_devices('GPU')}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: Adam default (0.001)\n")
        f.write(f"Final Train Accuracy: {train_acc*100:.2f}%\n")
        f.write(f"Final Val Accuracy: {val_acc*100:.2f}%\n")
    print("Saved summary to summary.txt")


# MAIN
def main():
    print("DATASET_PATH:", DATASET_PATH)

    print("Loading dataset...")
    X_train, y_train, X_test, y_test = load_split_data(DATASET_PATH)

    # Reshape for TF/Keras
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)

    print("Building Mini-Xception model...")
    model = build_mini_xception((IMG_SIZE[0], IMG_SIZE[1], 1), NUM_CLASSES)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Training model...")
    history = train_and_evaluate(model, X_train, y_train, X_test, y_test)

    # Load best saved model
    print("Loading best model from disk...")
    best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    # Final evaluation
    print("Evaluating...")
    y_pred = np.argmax(best_model.predict(X_test), axis=1)
    print(classification_report(y_test, y_pred, target_names=EMOTIONS))

    cm = confusion_matrix(y_test, y_pred)
    save_confusion_matrix(cm, EMOTIONS)

    # Save summary.txt
    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    write_summary(final_train_acc, final_val_acc)

    print("All outputs saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
