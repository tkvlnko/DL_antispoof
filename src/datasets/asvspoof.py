"""
src/datasets/asvspoof.py
~~~~~~~~~~~~~~~~~~~~~~~~
Датасет ASVspoof-2019 LA + «готовый» препроцесс аудио → лог-спектр.

⚑ Главные решения, откуда взялись
────────────────────────────────
1. STFT-фронтенд (n_fft=512, hop=160, win=400) — как в статье Speech
   Technology Center (DOI: 10.1109/ICASSP.2020.9053040). Всю цепочку
   torchaudio → |FFT|^2 → log(1+x).
2. Никакого trim / фиксированной длины — модель LightCNN-9 умеет работать
   с T любого размера, а AdaptiveAvgPool потом ужимает частотную ось.
3. Баланс классов «bona/spoof» регулируется аргументом `bonafide_ratio`;
   при значении 0.5 получаем примерно равные классы (совет из обзора
   EER-vs-loss функций: arXiv 2103.11326).
4. Допустимы *пропуски* файлов: если wav нет на диске —
   «спокойно» пропускаем строку (иначе тренировку ломает IndexError).
"""

from __future__ import annotations

from pathlib import Path
import random
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
import torch.nn.functional as F



class ASVspoofDataset(Dataset):
    # параметры STFT по умолчанию (можно переопределять в YAML)
    DEFAULT_N_FFT = 512
    DEFAULT_HOP   = 160
    DEFAULT_WIN   = 400
    SAMPLE_RATE   = 16_000      # датасет 16 кГц

    PROTO_MAP = {
        "train": "ASVspoof2019.LA.cm.train.trl.txt",
        "dev":   "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval":  "ASVspoof2019.LA.cm.eval.trl.txt",
    }

    def __init__(
        self,
        root: str | Path,
        split: str = "train",              # train / dev / eval
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP,
        win_length: int = DEFAULT_WIN,
        bonafide_ratio: float | None = None,  # 0.5 = баланс классов
        spec_aug: torch.nn.Module | None = None,  # аугментация (train)
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.n_fft, self.hop, self.win = n_fft, hop_length, win_length
        self.bonafide_ratio = bonafide_ratio
        self.spec_aug = spec_aug

        # ── Hann-окно создаётся один раз ───────────────────────
        self.window = torch.hann_window(self.win)

        # ── строим индекс (список словарей [{path, label}, …]) ─
        proto_path = self.root / "protocols" / self.PROTO_MAP[split]
        self.index: List[dict] = []
        miss = 0

        with proto_path.open() as f:
            for line in f:
                utt_id, wav_name, _, _, label = line.strip().split()
                wav_path = (
                    self.root
                    / f"ASVspoof2019_LA_{split}"
                    / "flac"
                    / f"{wav_name}.flac"
                )
                if not wav_path.exists():
                    miss += 1
                    continue
                target = 1 if label == "bonafide" else 0
                self.index.append({"path": wav_path, "label": target})

        if miss:
            print(f"[ASVspoofDataset] skipped {miss} missing wavs in {split}")

        # ── балансировка классов (только train) ────────────────
        if bonafide_ratio is not None and split == "train":
            self._balance_classes()

        print(f"[ASVspoofDataset] {split}: {len(self)} wavs "
              f"(self.bona_ratio={self.bonafide_ratio})")

    # ------------------------------------------------------------
    # Вспомогательные
    # ------------------------------------------------------------
    def _balance_classes(self):
        """
        Совмещаем bonafide и spoof списки с нужным соотношением.
        """
        bona = [it for it in self.index if it["label"] == 1]
        spoof = [it for it in self.index if it["label"] == 0]

        n_bona = int(len(self.index) * self.bonafide_ratio)
        n_spoof = len(self.index) - n_bona

        random.shuffle(bona)
        random.shuffle(spoof)
        self.index = bona[:n_bona] + spoof[:n_spoof]
        random.shuffle(self.index)

    def _wav_to_logspec(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Преобразуем сигнал -> лог-энергетическая спектрограмма.
        wav: Tensor [1, T]
        return: Tensor [freq, time]
        """
        # torchaudio >= 2.0 имеет комплекс-stft; возвращаемая форма F×T×2
        stft = torch.stft(
            wav.squeeze(),
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win,
            window=self.window,
            return_complex=True,
        )
        power = stft.abs().pow(2)
        logspec = torch.log1p(power)  # log(1+x) — устойч. при x≈0
        return logspec     # F×T

    # ------------------------------------------------------------
    # Реализация Dataset
    # ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rec = self.index[idx]
        wav, sr = torchaudio.load(rec["path"])
        if sr != self.SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, self.SAMPLE_RATE)

        # feat = self._wav_to_logspec(wav)    # F×T
                # ── (1) Случайный/центральный кроп перед STFT ─────────
        SEG_SEC = 4.0                          # длина сегмента
        seg_samples = int(SEG_SEC * self.SAMPLE_RATE)

        if self.split == "train":              # random crop
            if wav.shape[1] > seg_samples:
                start = random.randint(0, wav.shape[1] - seg_samples)
                wav = wav[:, start:start + seg_samples]
            elif wav.shape[1] < seg_samples:   # короткую — допаддить
                pad = seg_samples - wav.shape[1]
                wav = F.pad(wav, (0, pad))
        else:                                  # dev/eval — центр кроп
            if wav.shape[1] > seg_samples:
                start = (wav.shape[1] - seg_samples) // 2
                wav = wav[:, start:start + seg_samples]
            elif wav.shape[1] < seg_samples:
                pad = seg_samples - wav.shape[1]
                wav = F.pad(wav, (0, pad))

        feat = self._wav_to_logspec(wav)       # F×T  (теперь T ≈ 400)
 
        if self.spec_aug and self.split == "train":
            feat = self.spec_aug(feat)      # F×T (аугмент)

        wav_name = rec["path"].stem            # 'LA_E_1234567'
        return feat, rec["label"], wav_name 
         # collate → pad & unsqueeze
