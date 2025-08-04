import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import hydra
from hydra.utils import instantiate, get_original_cwd
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from src.datasets.asvspoof import ASVspoofDataset
from src.datasets.collate import pad_collate
from src.logger.cometml import CometMLWriter  # если ты инстанцируешь напрямую, иначе через hydra.utils.instantiate
from src.metrics.calculate_eer import compute_eer, roc_curve  # твоя функция: (bona, spoof) -> eer, thr, fpr, tpr

# main entrypoint через Hydra
@hydra.main(config_path="src/configs", config_name="lcnn_min", version_base=None)
def main(cfg: DictConfig):

    # --------- 1. подготовка ----------------------------------------------------------------
    device = torch.device(cfg.trainer.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    # логгер / CometML
    try:
        comet = instantiate(
            cfg.writer,
            logger=None,
            project_config=OmegaConf.to_container(cfg, resolve=True),
        )
    except Exception:
        # fallback: если напрямую
        comet = CometMLWriter(logger=None, project_config=OmegaConf.to_container(cfg, resolve=True),
                              project_name=cfg.writer.project_name if "project_name" in cfg.writer else "default",
                              workspace=cfg.writer.workspace if "workspace" in cfg.writer else None,
                              run_name=cfg.writer.run_name if "run_name" in cfg.writer else None,
                              mode=cfg.writer.mode if "mode" in cfg.writer else "online")

    # датасеты
    orig_root = Path(get_original_cwd())          # корень проекта
    train_root = orig_root / cfg.train.root
    dev_root   = orig_root / cfg.dev.root
    eval_root  = orig_root / cfg.eval.root


    train_ds = instantiate(cfg.train, root=str(train_root))
    dev_ds   = instantiate(cfg.dev,   root=str(dev_root))
    eval_ds  = instantiate(cfg.eval,  root=str(eval_root))

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



    # модель/оптимизатор/лосс/scheduler
    model = hydra.utils.instantiate(cfg.model).to(device)
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)

    # -------------------------------------------------------------------------------

    global_step = 0
    best_eer = float("inf")

    # Optional: сразу создаём панели в Comet для автоматического отображения
    try:
        comet.exp.log_other("create_chart", {"name": "Loss", "chart": "line", "metric_list": ["loss_train"]})
        comet.exp.log_other("create_chart", {"name": "EER", "chart": "line", "metric_list": ["eer_dev"]})
    except Exception:
        pass  # не критично

    # --------- 2. тренировочный цикл ---------------------------------------------------------
    for epoch in range(cfg.trainer.epochs):
        model.train()
        pbar = tqdm(train_dl, desc=f"epoch {epoch+1}", leave=False)
        running_loss = 0.0

        for batch_idx, batch in enumerate(pbar):
            # pad_collate даёт (x, y, lengths)
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch  # на случай, если вдруг другой collate
            x = x.to(device)           # B×1×F×T
            y = y.to(device)           # B×1 float для BCE

            logits = model(x)          # [B,1]
            loss = loss_fn(logits, y)  # BCEWithLogitsLoss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            running_loss += loss.item()

            # логируем каждый шаг
            comet.set_step(global_step, mode="train")
            comet.add_scalars({
                "loss": loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
            })

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # scheduler (после эпохи)
        try:
            scheduler.step()
        except Exception:
            pass  # некоторые schedulers вызываются вручную иначе

        # --------- 3. evaluation на dev внутри функции --------------------------------------
        eer, thr, fpr, tpr = evaluate_and_log(model, dev_dl, device, comet, epoch + 1, prefix="dev")
        eer_eval, thr_eval, *_ = evaluate_and_log(model, eval_dl, device, comet, epoch + 1 + 0.5, prefix="eval")


        # чекпойнт по лучшему eer
        if eer is not None and eer < best_eer:
            best_eer = eer
            # можно сохранять модель
            torch.save(model.state_dict(), os.path.join(os.getcwd(), "best_lightcnn.pt"))
            # логнуть сохранение
            comet.add_text("best_checkpoint", f"epoch={epoch+1}, eer={eer:.4%}, thr={thr:.4f}")

        # печать прогресса
        # print(f"Epoch {epoch+1} train_loss={running_loss/len(train_dl):.4f} eer_dev={eer if eer is not None else 'nan':.4f}")
        print(f"Epoch {epoch:2d} train_loss={running_loss/len(train_dl):.4f} "
              f"eer_dev={eer if eer is not None else 'nan':.4f}  eer_eval={eer_eval:.4%}")

    # окончание
    print(f"Finished. Best dev EER: {best_eer:.4%}")


# ------------- evaluate + логирование -----------------------------
@torch.no_grad()
def evaluate_and_log(model, loader, device, comet, epoch_step):
    model.eval()
    bona_scores, spoof_scores = [], []

    for x, y, *_ in loader:            # допускаем третий элемент (имя файла)
        x, y = x.to(device), y.to(device)

        logits = model(x)                            # (B, 2)
        prob_bona = torch.softmax(logits, 1)[:, 1]   # ↑ bona-confidence

        scores  = prob_bona.cpu().numpy()            # (B,)
        labels  = y.squeeze(-1).cpu().numpy()        # (B,)

        bona_scores .append(scores[labels == 1])
        spoof_scores.append(scores[labels == 0])

    if not bona_scores or not spoof_scores:
        return None, None, None, None

    bonafide_scores = np.concatenate(bona_scores)    # 1-D
    other_scores    = np.concatenate(spoof_scores)   # 1-D

    # --- EER ---
    eer, threshold = compute_eer(bonafide_scores, other_scores)

    # --- ROC ---
    from sklearn.metrics import roc_curve
    all_scores = np.concatenate((bonafide_scores, other_scores))
    all_labels = np.concatenate((np.ones_like(bonafide_scores),
                                 np.zeros_like(other_scores)))
    fpr, tpr, _ = roc_curve(all_labels, all_scores, pos_label=1)

    # --- Comet ---
    comet.set_step(epoch_step, mode=prefix)
    comet.add_scalar(f"eer_{prefix}", eer)
    comet.add_scalar(f"threshold_{prefix}", threshold)

    try:
        import pandas as pd
        comet.add_table("roc_curve", pd.DataFrame({"fpr": fpr, "tpr": tpr}))
    except ImportError:
        pass

    return eer, threshold, fpr, tpr


if __name__ == "__main__":
    main()