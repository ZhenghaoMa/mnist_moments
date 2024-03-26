import torch
import torch.nn as nn



class FCModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        hidden_dim = 2048
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            # nn.Dropout(0.4),
            nn.Linear(hidden_dim, 10)
        )

    def forward(self, x):
        # x = transforms.functional.normalize(x, mean=[0.5, 0.5, 0.5],
        #                                     std=[0.5, 0.5, 0.5])
        logits = self.fc(x)
        return logits

class Conv1dModel(nn.Module):
    def __init__(self, input_dim=106):
        super().__init__()
        self.input_dim = input_dim
        hidden_dim = 2048
        channel = 32

        self.conv1d = nn.Sequential(
            nn.Conv1d(1, channel, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(channel),
            nn.ReLU(True),
            nn.Conv1d(channel, channel*2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(channel*2),
            nn.ReLU(True),
            nn.Conv1d(channel*2, channel*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(channel*4),
            nn.ReLU(True),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(channel*4, 10)
        )

    def forward(self, x):
        # x = transforms.functional.normalize(x, mean=[0.5, 0.5, 0.5],
        #                                     std=[0.5, 0.5, 0.5])
        x = x.unsqueeze(1)
        logits = self.conv1d(x)
        return logits

