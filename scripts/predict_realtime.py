import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_arch import ASLClassifier
import cv2
import mediapipe as mp
import numpy as np
import torch
import json
import os
from model.model_arch import ASLClassifier

# CONFIG
SEQ_LEN = 50
ROI_BOX = (100, 100, 400, 400)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load labels
with open("labels.json") as f:
    label_map = json.load(f)
reverse_label_map = {v: k for k, v in label_map.items()}

# Load model
model = ASLClassifier(num_classes=len(label_map)).to(DEVICE)
model.load_state_dict(torch.load("models/asl_model.pt", map_location=DEVICE))
model.eval()

# Setup Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# Prediction state
seq_buffer = []
predicted_word = ""
prediction_done = False

# Hand in ROI check
def hand_in_roi(landmarks, width, height):
    for lm in landmarks.landmark:
        x, y = int(lm.x * width), int(lm.y * height)
        if not (ROI_BOX[0] < x < ROI_BOX[2] and ROI_BOX[1] < y < ROI_BOX[3]):
            return False
    return True

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    cv2.rectangle(frame, (ROI_BOX[0], ROI_BOX[1]), (ROI_BOX[2], ROI_BOX[3]), (0, 255, 255), 2)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if not prediction_done and hand_in_roi(hand_landmarks, width, height):
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                seq_buffer.append(coords)

                if len(seq_buffer) > SEQ_LEN:
                    seq_buffer = seq_buffer[-SEQ_LEN:]

                if len(seq_buffer) == SEQ_LEN:
                    input_tensor = torch.tensor([seq_buffer], dtype=torch.float32).to(DEVICE)
                    with torch.no_grad():
                        output = model(input_tensor)
                        pred = torch.argmax(output, dim=1).item()
                        predicted_word = reverse_label_map[pred]
                        prediction_done = True

    else:
        prediction_done = False
        seq_buffer.clear()

    if predicted_word and prediction_done:
        cv2.putText(frame, f"Signed: {predicted_word}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("ASL Realtime Prediction", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()