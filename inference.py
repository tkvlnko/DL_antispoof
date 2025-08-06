import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.metrics.calculate_eer import compute_eer
from sklearn.metrics import roc_curve

@torch.no_grad()
def evaluate_and_log(model, loader, device, comet, epoch_step, prefix="dev"):
    model.eval()
    bona_scores, spoof_scores = [], []

    pbar = tqdm(
        loader,
        desc=f"[{prefix.upper()}] Epoch {epoch_step:.1f}",
        leave=False,
        unit="batch",
    )

    for x, y, *_ in pbar:            
        x, y = x.to(device), y.to(device)

        logits = model(x)                            
        prob_bona = torch.softmax(logits, 1)[:, 1]   

        scores  = prob_bona.cpu().numpy()            
        labels  = y.squeeze(-1).cpu().numpy()        

        bona_scores .append(scores[labels == 1])
        spoof_scores.append(scores[labels == 0])

    if not bona_scores or not spoof_scores:
        return None, None, None, None

    bonafide_scores = np.concatenate(bona_scores)    
    other_scores    = np.concatenate(spoof_scores)   

    # --- EER ---
    eer, threshold = compute_eer(bonafide_scores, other_scores)

    # --- ROC ---
    all_scores = np.concatenate((bonafide_scores, other_scores))
    all_labels = np.concatenate((np.ones_like(bonafide_scores),
                                 np.zeros_like(other_scores)))
    fpr, tpr, _ = roc_curve(all_labels, all_scores, pos_label=1)

    # --- Comet ---
    comet.set_step(epoch_step, mode=prefix)
    comet.add_scalar(f"eer_{prefix}", eer)
    comet.add_scalar(f"threshold_{prefix}", threshold)

    try:
        comet.add_table("roc_curve", pd.DataFrame({"fpr": fpr, "tpr": tpr}))
    except ImportError:
        pass

    return eer, threshold, fpr, tpr
