import torch
import torch.nn as nn
import torch.nn.functional as F

from .DeepConvLSTM import DeepConvLSTM
from .AudioModel import AudioModel
from ..preprocessing.extract_all_features import humid_feature_dim

class MultiModalModel(nn.Module):
    def __init__(self, n_classes=16, lstm_hidden=128, imu_l=True, imu_r=True, audio=True, humid=True, emb=False):
        super().__init__()

        # store parts:
        self.imu_l = imu_l
        self.imu_r = imu_r
        self.humid = humid
        self.audio = audio

        self.emb = emb

        # ---- AUDIO BRANCH ----
        if audio:
            self.audio_model = AudioModel(n_classes=n_classes, lstm_hidden=lstm_hidden, emb=True)

        # ---- IMU BRANCH(ES) ----
        if imu_l or imu_r:
            self.imu_encoder = DeepConvLSTM(emb=True)

        # ---- FINAL CLASSIFIER ----
        fused_dim = int(audio) * 256 + 128 * int(imu_l) + 128 * int(imu_r) + int(humid) * humid_feature_dim
        self.fused_dim = fused_dim
        # audio(256) + left imu + right imu + humidity
        # audio(256) + left imu + right imu + humidity

        # self.humid_encoder = DeepConvLSTM(emb=True, input_channels=1)  ## not currently used

        self.fc_final = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def freeze_encoders(self, freeze=True):
        for param in self.parameters():
            param.requires_grad = not freeze

    def reset_classification_head(self, n_classes):
        print("Resetting MultiModalModel classification head to", n_classes, "classes")
        self.fc_final = nn.Sequential(
            nn.Linear(self.fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def enable_linear_probing(self, n_classes):
        self.fc_final = nn.Linear(self.fused_dim, n_classes)

    def forward(self, audio=None, imu_left=None, imu_right=None, humid=None):
        audio_feat = imu_l_feat = imu_r_feat = None
        # ---- AUDIO ----
        if self.audio:
            audio_feat = self.audio_model(audio)  # (B, 256)

        # ---- IMU ----
        if self.imu_l:
            imu_l_feat = self.imu_encoder(imu_left)   # (B, 64)
        if self.imu_r:
            imu_r_feat = self.imu_encoder(imu_right)  # (B, 64)

        if self.humid and humid is None:
            raise(Exception("humidity required, but not passed."))
        elif not self.humid:
            humid=None
        #else:
        #    humid = self.humid_encoder(torch.unsqueeze(humid, -1))

        # ---- FUSION ----
        fused = torch.cat([x for x in [audio_feat, imu_l_feat, imu_r_feat, humid] if x is not None], dim=1)
        if self.emb == True:
            return fused

        out = self.fc_final(fused)

        if self.emb  == "both":
            return out, fused
        return out