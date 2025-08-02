# train.py
import logging
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import comet_ml
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm


# ─── утилиты ───────────────────────────────────────────
from src.metrics.calculate_eer import compute_eer            # из файла calculate_eer.py
from src.logger.cometml import CometMLWriter     # ваш класс логгера
from src.datasets.collate import pad_collate 
from src.datasets.asvspoof import ASVspoofDataset  

# reproducibility
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)


@torch.inference_mode()
def evaluate(model, loader, device):
    """
    Прогон по dev-сплиту → eer, threshold.
    """
    model.eval()

    bona_scores, spoof_scores = [], []

    for x, y, _ in loader:           # y: float32 [B,1]
        x, y = x.to(device), y.to(device)

        logits = model(x)            # [B,1]
        prob   = torch.sigmoid(logits).squeeze(1).cpu()  # bonafide-prob

        bona_scores.extend(prob[y.squeeze(1) == 1].tolist())
        spoof_scores.extend(prob[y.squeeze(1) == 0].tolist())

    # если по какой-то причине в выборке нет обоих классов
    if len(bona_scores) == 0 or len(spoof_scores) == 0:
        return None, None

    eer, thr = compute_eer(np.array(bona_scores),
                           np.array(spoof_scores))
    return eer, thr


@hydra.main(config_path="src/configs", config_name="lcnn_min", version_base=None)
def main(cfg):
    # ─── логгер python + Comet ───────────────────────────────
    global_step = 0
    log = logging.getLogger("train")
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler())

    comet = CometMLWriter(
        log,
        OmegaConf.to_container(cfg, resolve=True),     # project_config
        project_name=cfg.writer.project_name,
        workspace=cfg.writer.workspace,
        run_name=cfg.writer.run_name,
        mode=cfg.writer.mode,
    )
    comet.add_text("yaml_config", OmegaConf.to_yaml(cfg))

    # ─── датасеты и лоадеры ─────────────────────────────────
    dev_ds = ASVspoofDataset(
        root="data/ASVspoof2019_LA",
        split="dev",
        n_fft=cfg.dataset.n_fft,
        hop_length=cfg.dataset.hop_length,
        win_length=cfg.dataset.win_length,
    )

    dev_dl = DataLoader(
        dev_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=pad_collate,          # та же функция
    )


    train_ds = instantiate(cfg.dataset)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        drop_last=cfg.drop_last,
        collate_fn=pad_collate,
    )

    dev_cfg = cfg.dataset.copy()
    dev_cfg["split"] = "dev"
    dev_cfg["bonafide_ratio"] = None
    dev_ds = instantiate(dev_cfg)
    dev_dl = DataLoader(dev_ds, batch_size=cfg.batch_size, shuffle=False)

    # ─── модель, оптимайзер, lr-sched ───────────────────────
    device = torch.device(cfg.trainer.device)
    model = instantiate(cfg.model).to(device)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
    loss_fn = instantiate(cfg.loss_fn)

    global_step = 0

    for epoch in range(cfg.trainer.epochs):
        running = 0.0
        pbar = tqdm(train_dl, desc=f"epoch {epoch+1}", ncols=80)

        for x, y, lengths in pbar:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            global_step += 1
            comet.set_step(global_step, mode="train")     # <─ ключевое
            comet.add_scalars({
                "loss": loss.item(),
                "lr":   scheduler.get_last_lr()[0],
            })

            # tqdm & comet
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            # if global_step % cfg.trainer.log_steps == 0:
            #     comet.set_step(global_step, mode="train")
            #     comet.add_scalar("loss", loss.item())

        scheduler.step()

        # ─── evaluation на dev ──────────────────────────────
        eer, thr = evaluate(model, dev_dl, device)

        if eer is not None:                     # всё ок
            comet.set_step(epoch + 1, mode="dev")      # «epoch» как step
            comet.add_scalars({
                "eer": eer,            # появится eer_dev
                "thr": thr,            # можно тоже отрисовать
            })
            log.info(f"[dev] epoch {epoch+1} | EER={eer:.4%} | thr={thr:.3f}")

    model.train() 
    comet.finish()
    log.info("Training done.")


if __name__ == "__main__":
    main()
