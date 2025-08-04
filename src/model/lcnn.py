# src/model/lightcnn_la.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MFM(nn.Module):
    """Max-Feature-Map (2/1) для conv и fc-слоёв :contentReference[oaicite:12]{index=12}"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=0, fc=False):
        super().__init__()
        if fc:
            self.filter = nn.Linear(in_ch, out_ch * 2, bias=True)
            self.split_dim = 1
        else:
            self.filter = nn.Conv2d(in_ch, out_ch * 2, k, s, p, bias=True)
            self.split_dim = 1
        self.out_ch = out_ch

    def forward(self, x):
        x = self.filter(x)
        a, b = torch.split(x, self.out_ch, dim=self.split_dim)
        return torch.max(a, b)          # competitive activation


def nin(in_ch, out_ch):                 # 1×1 conv + MFM
    return MFM(in_ch, out_ch, k=1, s=1, p=0)


class LightCNN9(nn.Module):
    """
    LightCNN-9 под голосовой антиспуфинг.
    Вход: B×1×F×T, где F=257 (|FFT|/2+1), T — любая длина.
    Выход: логиты (B×n_classes).
    """
    def __init__(self, n_classes: int = 2, dropout_p: float = 0.2):
        super().__init__()

        self.features = nn.Sequential(
            # ------- блок 1 -------
            MFM( 1,  48, k=5, p=2),            # Conv1 → MFM1
            nn.MaxPool2d(2, 2),                # 1/2 по частоте и времени

            # ------- блок 2 -------
            nin(48,  48),                      # 1×1 conv (Conv2a)
            MFM(48, 96, k=3, p=1),             # Conv2
            nn.MaxPool2d(2, 2),

            # ------- блок 3 -------
            nin(96,  96),
            MFM(96, 192, k=3, p=1),
            nn.MaxPool2d(2, 2),

            # ------- блок 4 -------
            nin(192, 192),
            MFM(192, 128, k=3, p=1),

            # ------- блок 5 -------
            nin(128, 128),
            MFM(128, 128, k=3, p=1),
            nn.MaxPool2d(2, 2),
        )

        # усредняем только по частотной оси, время оставляем — average pooling рекомендуется в :contentReference[oaicite:13]{index=13}
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))   # B×C×1×T'

        self.classifier = nn.Sequential(
            nn.Flatten(1),                    # вектор 128
            MFM(128, 256, fc=True),          # fc1 + MFM
            nn.Dropout(dropout_p),
            nn.Linear(256, 2)        # bonafide-логит, spoof-логит
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
