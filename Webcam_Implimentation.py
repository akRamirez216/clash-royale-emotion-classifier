import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from trained_resnet_model import ResNet18_emotion
import imageio
import pygame
import time

# Initialize Audio
try:
    pygame.mixer.init()
except Exception as e:
    print("Warning: pygame.mixer.init() failed:", e)

# Emote File Paths
GIF_PATHS = {
    "angry": "gifs/angry.gif",
    "disgusted": "gifs/disgusted.gif",
    "fearful": "gifs/fearful.gif",
    "happy": "gifs/happy.gif",
    "neutral": "gifs/neutral.gif",
    "sad": "gifs/sad.gif",
    "surprised": "gifs/surprised.gif",
    "unknown": "gifs/unknown.gif",
}

SOUND_PATHS = {
    "angry": "sounds/angry.mp3",
    "disgusted": "sounds/disgusted.mp3",
    "fearful": "sounds/fearful.mp3",
    "happy": "sounds/happy.mp3",
    "neutral": "sounds/neutral.mp3",
    "sad": "sounds/sad.mp3",
    "surprised": "sounds/surprised.mp3",
    "unknown": "sounds/unknown.mp3",
}

# Device Setup
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)

# Load Model
MODEL_PATH = "trained_model/best_emotion_resnet18.pth"
model = ResNet18_emotion(num_classes=7)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

EMOTION_LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

# Preprocessing Function
def preprocess(face_img):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img.astype("float32") / 255.0
    face_tensor = torch.tensor(face_img).unsqueeze(0).unsqueeze(0)
    face_tensor = (face_tensor - 0.485) / 0.229
    return face_tensor.to(device)

# Media Pipe Face Detection Setup
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Preload GIFS
GIF_FRAMES = {}
GIF_INDEX = {}
GIF_LAST_TIME = {}
GIF_DELAY = 0.06  # ~16 FPS
GIF_DISPLAY_SIZE = (240, 240)

for emotion, path in GIF_PATHS.items():
    try:
        gif = imageio.mimread(path)
        frames = []
        for f in gif:
            # Remove alpha channel if exists
            frame = f[:, :, :3] if f.shape[2] > 3 else f
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_resized = cv2.resize(frame_bgr, GIF_DISPLAY_SIZE, interpolation=cv2.INTER_AREA)
            frames.append(frame_resized)
        GIF_FRAMES[emotion] = frames if frames else None
        GIF_INDEX[emotion] = 0
        GIF_LAST_TIME[emotion] = 0.0
        print(f"Loaded GIF for {emotion}, {len(frames)} frames")
    except Exception as e:
        print(f"Failed to load GIF {path}: {e}")
        GIF_FRAMES[emotion] = None
        GIF_INDEX[emotion] = 0
        GIF_LAST_TIME[emotion] = 0.0

# Emote Cooldown
last_played = {"emotion": None, "time": 0}

# Play Sound
def play_sound(path):
    try:
        pygame.mixer.music.stop()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Failed to play sound {path}: {e}")

# GIF Display Settings
GIF_DISPLAY_SIZE = (300, 300)  # Bigger GIF
GIF_POSITION = (10, 10)        # Overlay position
UNKNOWN_THRESHOLD = 0.20
BUFFER_SIZE = 4                 # How many past frames to consider for smoothing
COOLDOWN = 1.0                  # Seconds before switching to a new emotion
emotion_buffer = []

# Main Webcam Loop
cv2.namedWindow("Emotion Recognition", cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)
print("Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    detected_emotion = None
    detected_conf = 0.0

    # Face Detection and Emotion Prediction
    if results.detections:
        det = results.detections[0]  # only first face
        bbox = det.location_data.relative_bounding_box
        ih, iw, _ = frame.shape
        x1 = max(int(bbox.xmin * iw), 0)
        y1 = max(int(bbox.ymin * ih), 0)
        w = max(int(bbox.width * iw), 1)
        h = max(int(bbox.height * ih), 1)

        face = frame[y1:y1+h, x1:x1+w]
        if face.size != 0:
            face_tensor = preprocess(face)
            with torch.no_grad():
                logits = model(face_tensor)
                probs = F.softmax(logits, dim=1)[0]
            pred_idx = probs.argmax().item()
            conf = probs[pred_idx].item()
            detected_emotion = "unknown" if conf < UNKNOWN_THRESHOLD else EMOTION_LABELS[pred_idx]
            detected_conf = conf

            # draw bounding box + label
            label_text = f"{detected_emotion} ({detected_conf:.2f})"
            cv2.putText(frame, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0,255,0), 2)

    # Buffer + majority vote: stores the last N detected emotions and chooses the most frequent one.
    # This helps smooth out rapid fluctuations and prevents the displayed emotion from switching too quickly.
    if detected_emotion:
        emotion_buffer.append(detected_emotion)
        if len(emotion_buffer) > BUFFER_SIZE:
            emotion_buffer.pop(0)

        # majority vote
        most_common = max(set(emotion_buffer), key=emotion_buffer.count)
    else:
        most_common = last_played["emotion"]  # keep previous if no detection

    now = time.time()

    # Switch emotion if buffer agrees and cooldown passed
    if most_common != last_played["emotion"] and (now - last_played["time"] > COOLDOWN):
        last_played["emotion"] = most_common
        last_played["time"] = now

        # play sound
        sound_path = SOUND_PATHS.get(most_common)
        if sound_path:
            play_sound(sound_path)

        # reset GIF animation
        if GIF_FRAMES.get(most_common):
            GIF_INDEX[most_common] = 0
            GIF_LAST_TIME[most_common] = now

    # Overlay GIF
    current_emotion = last_played["emotion"]
    if current_emotion and GIF_FRAMES.get(current_emotion):
        frames = GIF_FRAMES[current_emotion]
        idx = GIF_INDEX[current_emotion]
        last_t = GIF_LAST_TIME.get(current_emotion, 0.0)

        if (now - last_t) >= GIF_DELAY:
            idx = (idx + 1) % len(frames)
            GIF_INDEX[current_emotion] = idx
            GIF_LAST_TIME[current_emotion] = now

        gif_frame = frames[idx]
        gif_frame_resized = cv2.resize(gif_frame, GIF_DISPLAY_SIZE, interpolation=cv2.INTER_AREA)
        gh, gw, _ = gif_frame_resized.shape
        x_offset, y_offset = GIF_POSITION

        # safe overlay
        frame[y_offset:y_offset+gh, x_offset:x_offset+gw] = gif_frame_resized

    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()