'''
    Predict scores of the model
'''

import csv, torch
from torch.utils.data import DataLoader
from pathlib import Path
from src.datasets.asvspoof import ASVspoofDataset
from src.datasets.collate import pad_collate
from src.model.lcnn import LightCNN9   

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT   = Path("data/ASVspoof2019_LA")     
CKPT   = Path("best_lightcnn.pt")      
BATCH  = 64

# ============= dataset
ds = ASVspoofDataset(root=ROOT, split="eval")
dl = DataLoader(ds, batch_size=BATCH, shuffle=False,
                num_workers=4, collate_fn=pad_collate)

# =============  model
model = LightCNN9(n_classes=2)          
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval().to(DEVICE)

scores = {}
with torch.no_grad():
    for x, _, _, keys in dl:           
        x = x.to(DEVICE)
        prob_bona = torch.softmax(model(x), 1)[:, 1]   
        for k, s in zip(keys, prob_bona.cpu().numpy()):
            scores[k] = float(s)

username = "tsibragimova"                  
out = Path(f"students_solutions/{username}.csv")
with out.open("w", newline="") as f:
    writer = csv.writer(f)
    for k, s in scores.items():
        writer.writerow([k, s])

print(f"Saved {len(scores)} scores to {out}")
