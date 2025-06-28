# dataset/create_dataset.py

import os
import json

def create_label_map(data_path='data', output_path='dataset/label_map.json'):
    labels = sorted(os.listdir(data_path))  # Sort alphabetically
    label_map = {label: idx for idx, label in enumerate(labels)}

    with open(output_path, 'w') as f:
        json.dump(label_map, f, indent=4)

    print(f"✅ Label map created with {len(label_map)} classes.")

if __name__ == "__main__":
    create_label_map()

import cv2
import numpy as np
from tqdm import tqdm

# Load label map
with open('dataset/label_map.json', 'r') as f:
    label_map = json.load(f)

def extract_frames(video_path, num_frames=8, size=(112, 112)):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total < num_frames or not cap.isOpened():
        cap.release()
        return None

    frame_idxs = np.linspace(0, total - 1, num_frames).astype(int)
    frames = []

    for i in range(total):
        ret, frame = cap.read()
        if i in frame_idxs:
            if not ret:
                return None
            frame = cv2.resize(frame, size)
            frames.append(frame)

    cap.release()
    return np.array(frames)

def build_dataset(data_path='data'):
    X, y = [], []

    for label_name in tqdm(os.listdir(data_path), desc="Processing Words"):
        label_folder = os.path.join(data_path, label_name)
        label_idx = label_map[label_name]

        for video_file in os.listdir(label_folder):
            video_path = os.path.join(label_folder, video_file)
            frames = extract_frames(video_path)

            if frames is not None:
                X.append(frames)
                y.append(label_idx)

    X = np.array(X)
    y = np.array(y)

    np.save('dataset/X.npy', X)
    np.save('dataset/y.npy', y)

    print(f"✅ Dataset saved: {X.shape}, {y.shape}")

if __name__ == "__main__":
    create_label_map()  # keep this here
    build_dataset()

