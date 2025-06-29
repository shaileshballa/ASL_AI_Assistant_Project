import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model_arch import ASLClassifier
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from model.model_arch import ASLClassifier

# CONFIG
LANDMARK_DIR = "C:/Users/Dell/Documents/ASL_AI_Assistant_Project/data/landmark"
SEQ_LEN = 50
BATCH_SIZE = 16
EPOCHS = 75
LR = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create label map
class_names = sorted(os.listdir(LANDMARK_DIR))
label_map = {name: idx for idx, name in enumerate(class_names)}
reverse_label_map = {idx: name for name, idx in label_map.items()}
with open("labels.json", "w") as f:
    json.dump(label_map, f)

# Padding helper
def pad_sequence(seq, max_len=SEQ_LEN):
    if len(seq) > max_len:
        return seq[:max_len]
    else:
        padding = np.zeros((max_len - len(seq), 63))
        return np.vstack((seq, padding))

# Dataset
class ASLDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for label in os.listdir(root_dir):
            folder = os.path.join(root_dir, label)
            for file in os.listdir(folder):
                if file.endswith(".npy"):
                    path = os.path.join(folder, file)
                    self.samples.append((path, label_map[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        seq = np.load(path)
        padded = pad_sequence(seq)
        return torch.tensor(padded, dtype=torch.float32), torch.tensor(label)

# Load data
dataset = ASLDataset(LANDMARK_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Train
model = ASLClassifier(num_classes=len(label_map)).to(DEVICE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    total_loss = 0
    model.train()
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/asl_model.pt")
print("Model saved.")