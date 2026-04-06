import torch.nn as nn
import torch.nn.functional as F

class AudioModel(nn.Module):
    def __init__(self, n_classes=10, lstm_hidden=128, lstm_layers=1, emb=False):
        super(AudioModel, self).__init__()

        self.emb = emb

        # CNN feature extractor
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Reduce frequency dimension only, keep time dimension flexible
        self.freq_pool = nn.AdaptiveAvgPool2d((4, None))  # (freq -> 4, time unchanged)

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128 * 4,  # channels * freq_bins_after_pool
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Classifier
        self.fc1 = nn.Linear(lstm_hidden * 2, 128)  # *2 because bidirectional
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        """
        Accepts:
        - (B, n_mels, T)
        - (B, 1, n_mels, T)
        """
        # Ensure channel dimension exists
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, n_mels, T)

        # CNN
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # (B, 32, ...)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # (B, 64, ...)
        x = F.relu(self.bn3(self.conv3(x)))              # (B, 128, ...)

        # Pool frequency only
        x = self.freq_pool(x)  # (B, 128, 4, T')

        # Prepare for LSTM
        B, C, Freq, Time = x.shape
        x = x.permute(0, 3, 1, 2)        # (B, T, C, Freq)
        x = x.reshape(B, Time, C * Freq) # (B, T, features)

        # LSTM
        x, _ = self.lstm(x)  # (B, T, 2*lstm_hidden)

        # Temporal aggregation (mean over time)
        x = x.mean(dim=1)

        if self.emb:
            return x

        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x