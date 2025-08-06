import os
import torch
import hydra

import numpy as np
from tqdm import tqdm
from pathlib import Path

from torch.utils.data import DataLoader
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig, OmegaConf

from src.datasets.asvspoof import ASVspoofDataset
from src.datasets.collate import pad_collate
from src.logger.cometml import CometMLWriter  
from src.metrics.calculate_eer import compute_eer, roc_curve  
from inference import evaluate_and_log

@hydra.main(config_path="src/configs", config_name="lcnn9", version_base=None)
def main(cfg: DictConfig):

    device = torch.device(cfg.trainer.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

# ===================== logger
    try:
        comet = instantiate(
            cfg.writer,
            logger=None,
            project_config=OmegaConf.to_container(cfg, resolve=True),
        )
    except Exception:
        comet = CometMLWriter(logger=None, project_config=OmegaConf.to_container(cfg, resolve=True),
                            project_name=cfg.writer.project_name if "project_name" in cfg.writer else "default",
                            workspace=cfg.writer.workspace if "workspace" in cfg.writer else None,
                            run_name=cfg.writer.run_name if "run_name" in cfg.writer else None,
                            mode=cfg.writer.mode if "mode" in cfg.writer else "online")
    try:
        comet.exp.log_other("create_chart", {"name": "Loss", "chart": "line", "metric_list": ["loss_train"]})
        comet.exp.log_other("create_chart", {"name": "EER", "chart": "line", "metric_list": ["eer_dev"]})
    except Exception:
        pass  

# ===================== datasets
    orig_root = Path(get_original_cwd())          
    train_root = orig_root / cfg.train.root
    dev_root   = orig_root / cfg.dev.root
    eval_root  = orig_root / cfg.eval.root


    train_ds = instantiate(cfg.train, root=str(train_root))
    dev_ds   = instantiate(cfg.dev,   root=str(dev_root))
    eval_ds  = instantiate(cfg.eval,  root=str(eval_root))

# ===================== dataloader
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        collate_fn=pad_collate,
    )

    dev_dl = DataLoader(
        dev_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=pad_collate,
    )

    eval_dl = DataLoader(
        eval_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=pad_collate,
    )

# ===================== model/optimizer/loss/scheduler
    model = hydra.utils.instantiate(cfg.model).to(device)
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)


    global_step = 0
    best_eer = float("inf")

# ===================== training loop
    for epoch in range(cfg.trainer.epochs):
        model.train()
        pbar = tqdm(train_dl, desc=f"[TRAIN] Epoch {epoch+1}", leave=False)
        running_loss = 0.0

        for batch_idx, batch in enumerate(pbar):
            x, y, *rest = batch
            x = x.to(device)           # B×1×F×T
            y = y.to(device)           # B×1 float for BCE

            logits = model(x)          # [B,1]
            loss = loss_fn(logits, y)  # BCEWithLogitsLoss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            running_loss += loss.item()

            comet.set_step(global_step, mode="train")
            comet.add_scalars({
                "loss": loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
            })

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        try:
            scheduler.step()
        except Exception:
            pass  

# ===================== evaluation and logging
        eer, thr, fpr, tpr = evaluate_and_log(model, dev_dl, device, comet, epoch + 1, prefix="dev")
        eer_eval, thr_eval, *_ = evaluate_and_log(model, eval_dl, device, comet, epoch + 1 + 0.5, prefix="eval")


# ===================== checkpoint 
        if eer is not None and eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), os.path.join(os.getcwd(), "best_lightcnn.pt"))
            comet.add_text("best_checkpoint", f"epoch={epoch+1}, eer={eer:.4%}, thr={thr:.4f}")


        print(f"Epoch {epoch+1} train_loss={running_loss/len(train_dl):.4f} "
              f"eer_dev={eer if eer is not None else 'nan':.4f}  eer_eval={eer_eval:.4%}")

    print(f"Finished. Best dev EER: {best_eer:.4%}")


if __name__ == "__main__":
    main()