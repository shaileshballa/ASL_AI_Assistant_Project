# model/model_arch.py
import torch
import torch.nn as nn

class ASLClassifier(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_layers=2, num_classes=16):
        super(ASLClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        out = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        return self.fc(out)