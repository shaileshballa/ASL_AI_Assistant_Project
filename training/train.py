# training/train.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# training/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model.model_arch import ASL3DCNN
import os

# Custom dataset class
class ASLDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32) / 255.0  # Normalize to 0-1
        y = self.y[idx]
        return torch.tensor(x), torch.tensor(y, dtype=torch.long)  # Fixed type!

def train_model():
    # Load dataset
    print("📦 Loading dataset...")
    X = np.load("dataset/X.npy")
    y = np.load("dataset/y.npy")

    dataset = ASLDataset(X, y)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Training on: {device}")
    model = ASL3DCNN(num_classes=len(set(y))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss, correct = 0.0, 0

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == batch_y).sum().item()

        accuracy = correct / len(dataset)
        print(f"📚 Epoch {epoch+1}/{epochs} - Loss: {running_loss:.4f} - Accuracy: {accuracy:.4f}")

    # Save model
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/asl_3dcnn.pth")
    print("✅ Model saved to model/asl_3dcnn.pth")

if __name__ == "__main__":
    train_model()
