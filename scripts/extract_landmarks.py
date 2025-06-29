import os
import cv2
import numpy as np
import mediapipe as mp

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

# Path setup (relative from scripts/)
RAW_VIDEO_DIR = "C:/Users/Dell/Documents/ASL_AI_Assistant_Project/data/raw_videos"
LANDMARK_DIR = "C:/Users/Dell/Documents/ASL_AI_Assistant_Project/data/landmark"

def extract_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    landmark_seq = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            frame_landmarks = []
            for lm in hand.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
            landmark_seq.append(frame_landmarks)

    cap.release()
    return np.array(landmark_seq)

# Process all videos
for word in os.listdir(RAW_VIDEO_DIR):
    word_path = os.path.join(RAW_VIDEO_DIR, word)
    output_path = os.path.join(LANDMARK_DIR, word)
    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(word_path):
        if filename.endswith((".mp4", ".mov", ".avi")):
            video_path = os.path.join(word_path, filename)
            save_path = os.path.join(output_path, filename.replace(".mp4", ".npy"))

            print(f"Processing {video_path}...")
            landmarks = extract_landmarks_from_video(video_path)

            if len(landmarks) > 0:
                np.save(save_path, landmarks)
                print(f"✅ Saved: {save_path}")
            else:
                print(f"❌ No hand detected in: {video_path}")
