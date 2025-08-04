import torch
import torch.nn as nn
import random

class SpecAugment(nn.Module):
    """
    Простая реализация SpecAugment (frequency & time masking) на лог-спектрограмме.
    Ожидает вход: Tensor [F, T], возвращает тот же shape.
    Параметры:
        freq_mask_param: максимальная ширина frequency-маски
        time_mask_param: максимальная ширина time-маски
        num_freq_masks: сколько frequency-масок применять подряд
        num_time_masks: сколько time-масок применять подряд
        p: вероятность применения аугментации (иначе пропускается)
    """
    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 40,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        p: float = 1.0,
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.p = p

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        spec: [F, T], лог-спектрограмма (обычно log(1+|STFT|^2))
        """
        if not self.training:
            return spec
        if random.random() > self.p:
            return spec

        spec = spec.clone()  # не мутируем исходный

        freq_len, time_len = spec.shape

        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            if f == 0:
                continue
            f0 = random.randint(0, max(0, freq_len - f))
            spec[f0 : f0 + f, :] = 0.0

        # Time masking
        for _ in range(self.num_time_masks):
            t = random.randint(0, self.time_mask_param)
            if t == 0:
                continue
            t0 = random.randint(0, max(0, time_len - t))
            spec[:, t0 : t0 + t] = 0.0

        return spec
