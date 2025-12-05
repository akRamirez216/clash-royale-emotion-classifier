""""A emotion recognition project that extracts facial 
landmarks with Mini-Xception model and classifies them based on training 
data from the dataset. The system categorizes emotions into seven emotions, 
triggering matching Clash Royale emotes and sounds."""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.layers import Input, SeparableConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model


# Emotion categories
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
IMG_SIZE = (64, 64)  # Mini-Xception default
NUM_CLASSES = len(EMOTIONS)
BATCH_SIZE = 32
EPOCHS = 50
DATASET_PATH = "ananthu017/emotion-detection-fer"



def build_mini_xception(input_shape, num_classes):
    """Builds a proper Mini-Xception architecture."""
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


def load_data(dataset_path):
    """Loads images from the FER folders."""
    data = []
    labels = []

    for emotion in EMOTIONS:
        folder = os.path.join(dataset_path, emotion)
        for img_name in os.listdir(folder):
            path = os.path.join(folder, img_name)

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, IMG_SIZE)
            img = img.astype("float32") / 255.0

            data.append(img)
            labels.append(EMOTIONS.index(emotion))

    return np.array(data), np.array(labels)


def main():

    # Load and preprocess data
    data, labels = load_data(DATASET_PATH)
    data = np.expand_dims(data, -1)

    
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )

    # Build and train model
    model = build_mini_xception((IMG_SIZE[0], IMG_SIZE[1], 1), NUM_CLASSES)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test)
    )

    # Evaluate the model
    y_pred = np.argmax(model.predict(X_test), axis=1)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=EMOTIONS))

    model.save("mini_xception_emotion_model.keras")
    print("Saved model to mini_xception_emotion_model.keras")



if __name__ == "__main__":
    main()
