import torch
from trained_resnet_model import load_model, predict_emotion

# Load model
model = load_model("results/best_emotion_resnet18.pth")

# Load face
face = torch.load("face_tensor_preprocessed.pt")  # shape: (1,48,48)

# Predict with threshold 0.40
label, conf = predict_emotion(model, face, unknown_threshold=0.40)

print(label, conf)
