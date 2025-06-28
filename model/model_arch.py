# model/model_arch.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ASL3DCNN(nn.Module):
    def __init__(self, num_classes=16):
        super(ASL3DCNN, self).__init__()
        
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d((1, 2, 2))  # (T, H, W)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d((2, 2, 2))

        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.AdaptiveAvgPool3d((1, 1, 1))  # Global pooling

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, T, H, W, C) → (B, C, T, H, W)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        return self.fc(x)
