""""A emotion recognition project that extracts facial 
landmarks with Mini-Xception model and classifies them based on training 
data from the dataset. The system categorizes emotions into seven emotions, 
triggering matching Clash Royale emotes and sounds."""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, SeparableConv2D, BatchNormalization,
    ReLU, GlobalAveragePooling2D, Dense
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping



# CONSTANTS
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
DATASET_PATH = kagglehub.dataset_download("ananthu017/emotion-detection-fer")
MODEL_SAVE_PATH = 'best_emotion_model.keras'

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = len(EMOTIONS)



# MODEL: Mini-Xception
def build_mini_xception(input_shape, num_classes):
    """Builds the Mini-Xception model architecture."""
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

    model = Model(inputs, outputs)
    return model



# DATA LOADING
def load_data(dataset_path):
    data = []
    labels = []

    for emotion in EMOTIONS:
        emotion_dir = os.path.join(dataset_path, emotion)
        if not os.path.isdir(emotion_dir):
            print(f"Warning: Missing folder: {emotion_dir}")
            continue

        for filename in os.listdir(emotion_dir):
            fpath = os.path.join(emotion_dir, filename)

            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, IMG_SIZE)
            img = img.astype("float32") / 255.0

            data.append(img)
            labels.append(EMOTIONS.index(emotion))

    return np.array(data), np.array(labels)




# TRAINING FUNCTION
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """Trains the model and evaluates it on the test set."""

    # Model checkpoint to save the best model
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, early_stop]
    )

    # Plot Accuracy 
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Loss 
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

    return history



# CONFUSION MATRIX HEATMAP
def plot_confusion_matrix(y_true, y_pred, labels):
    """Plots the confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()



# MAIN FUNCTION
def main():

    # Load data
    print("Loading data...")
    data, labels = load_data(DATASET_PATH)

    # Check if data is loaded
    if len(data) == 0:
        print("No data found. Check DATASET_PATH.")
        return

    # Reshape data for model input
    data = np.expand_dims(data, -1)

    # Split dataset
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )

    # Build model  
    print("Building model...")
    model = build_mini_xception((IMG_SIZE[0], IMG_SIZE[1], 1), NUM_CLASSES)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train and evaluate model
    print("Training...")
    train_and_evaluate(model, X_train, y_train, X_test, y_test)

    # Load best model for final evaluation
    print("Loading best saved model...")
    best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)


    # Final evaluation
    print("Evaluating best model...")
    y_pred = np.argmax(best_model.predict(X_test), axis=1)

    print(classification_report(y_test, y_pred, target_names=EMOTIONS))

    print("Plotting confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, EMOTIONS)

    print("Best model saved at:", MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()
