import torch
import torch.nn.functional as F
from typing import List, Tuple

def pad_collate(batch):
    feats, labels = zip(*batch)
    feats = torch.stack(feats).unsqueeze(1)        # B×1×F×400
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    lengths = torch.full((len(feats),), feats[0].shape[-1])  # =400
    return feats, labels, lengths
