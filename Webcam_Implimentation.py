"""
Real-time facial emotion recognition using a pretrained ResNet model.

This script captures webcam video, detects faces with MediaPipe Face Detection,
crops and preprocesses each face to match the model's 48x48 grayscale input, and
uses the trained ResNet to classify the emotion into one of seven categories.
The predicted emotion is displayed alongside the face bounding box.
"""


import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F

# Load the pretrained ResNet model for emotion recognition
MODELNAME = 'resnet_emotion_model.pth' # Update with your model path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load(MODELNAME, map_location=device)
model.eval().to(device)

# Emotion labels
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# preprocess function to prepare face images for the model
def preprocess(face_img):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img.astype('float32') / 255.0

    # covert to PyTorch tensor
    face_tensor = torch.tensor(face_img).unsqueeze(0)  # Shape: (1, 48, 48)

    face_tensor = (face_tensor - 0.485) / 0.229  # Normalize
    
    return face_tensor.unsqueeze(0).to(device)  # Shape: (1, 1, 48, 48)


# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Start video capture
cap = cv2.VideoCapture(0)

print("Starting webcam. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x1 = int(bboxC.xmin * iw)
            y1 = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            x1 = max(0, x1)
            y1 = max(0, y1)

            # Crop the face region
            face_crop = frame[y1:y1+h, x1:x1+w]
            if face_crop.size == 0:
                continue

            # Preprocess the face image
            face_tensor = preprocess(face_crop)

            # Predict emotion
            with torch.no_grad():
                outputs = model(face_tensor)
                _, predicted = torch.max(outputs, 1)
                emotion = EMOTION_LABELS[predicted.item()]

            # impliment popup for clash royale emote:

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break








