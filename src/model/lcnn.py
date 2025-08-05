import torch
import torch.nn as nn
import torch.nn.functional as F

class MFM(nn.Module):
    """
    Max-Feature-Map (2 channels -> 1) for conv & FC-layers 
    - in_ch: input channels
    - out_ch: output channels (after MFM)
    - k, s, p: kernel size, stride, padding (for Conv2d)
    """
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
        return torch.max(a, b)          


# network-in-network: 1×1 conv + MFM
def nin(in_ch, out_ch):                 
    return MFM(in_ch, out_ch, k=1, s=1, p=0)


class LightCNN9(nn.Module):
    def __init__(self, n_classes: int = 2, dropout_p: float = 0.2):
        super().__init__()

        self.features = nn.Sequential(
            # ------- block 1 -------           # Conv1: 128×128×96 ->
            MFM(1,  48, k=5, p=2),              # MFM1: 128×128×48->
            nn.MaxPool2d(2, 2),                 # Pool1: 64×64×48

            # ------- block 2 -------
            nin(48,  48),                       # Conv2a: 64×64×96 -> MFM2a: 64×64×48
            MFM(48, 96, k=3, p=1),              # Conv2: 64×64×192 -> MFM2: 64×64×96
            nn.MaxPool2d(2, 2),                 # Pool2: 32×32×96

            # ------- block 3 -------
            nin(96,  96),                       # Conv3a: 32×32×192 -> MFM3a: 32×32×96
            MFM(96, 192, k=3, p=1),             # Conv3: 32×32×384 -> MFM3: 32×32×192
            nn.MaxPool2d(2, 2),                 # Pool3: 16×16×192

            # ------- block 4 -------
            nin(192, 192),                      # Conv4a: 16×16×384 -> MFM4a: 16×16×192
            MFM(192, 128, k=3, p=1),            # Conv4: 16×16×256 -> MFM4: 16×16×128

            # ------- block 5 -------
            nin(128, 128),                      # Conv5a: 16×16×256 -> MFM5a: 16×16×128
            MFM(128, 128, k=3, p=1),            # Conv5: 16×16×256 -> MFM5: 16×16×128
            nn.MaxPool2d(2, 2),                 # Pool5: 8×8×128
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))   # 8×8×128 -> 1×1×128

        self.classifier = nn.Sequential(
            nn.Flatten(1),                    # 1×1×128 -> 128
            MFM(128, 256, fc=True),           # fc1 + MFM
            nn.Dropout(dropout_p),
            nn.Linear(256, 2)                 # bonafide-logit, spoof-logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
