import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepConvLSTM(nn.Module):
    def __init__(self, input_channels=6, hidden_size=128, emb=True):
        super().__init__()

        # Conv layers (temporal)
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=5, padding=2)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=False,

        )

    def forward(self, x):
        """
        x: (B, T, 6)
        """
        x = x.permute(0, 2, 1)  # (B, 6, T)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.permute(0, 2, 1)  # (B, T, 128)

        x, _ = self.lstm(x)  # (B, T, 2*hidden)
        x = x[:, -1, :]  # use last value in lstm output, no attention

        return x

# In[131]:
