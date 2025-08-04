import torch
import random
import matplotlib.pyplot as plt
from src.datasets.spec_augment import SpecAugment

# фиксация для детерминизма
random.seed(0)
torch.manual_seed(0)

# синтетическая спектрограмма: 257 частотных бин и 400 тайм-фреймов
dummy = torch.randn(257, 400)

# создаём аугментер
aug = SpecAugment(
    freq_mask_param=30,
    time_mask_param=80,
    num_freq_masks=2,
    num_time_masks=2,
    p=1.0,
)

# убедимся, что в train режиме
print("before train(): training flag =", aug.training)
aug.train()
print("after train(): training flag =", aug.training)

# применяем
augmented = aug(dummy)

# сравнение
diff = dummy - augmented
zeroed_elements = (augmented == 0.0).sum().item()
total_elements = augmented.numel()
same_all = torch.equal(dummy, augmented)

print(f"All equal (no change)? {same_all}")
print(f"Zeroed elements: {zeroed_elements}/{total_elements} ({zeroed_elements/total_elements:.4f})")
print(f"Diff abs mean: {diff.abs().mean().item():.6f}, max: {diff.abs().max().item():.6f}")

# визуализация
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(dummy.numpy(), aspect="auto", origin="lower")
plt.colorbar(shrink=0.5)

plt.subplot(1, 3, 2)
plt.title("Augmented")
plt.imshow(augmented.numpy(), aspect="auto", origin="lower")
plt.colorbar(shrink=0.5)

plt.subplot(1, 3, 3)
plt.title("Absolute Diff")
plt.imshow(diff.abs().numpy(), aspect="auto", origin="lower")
plt.colorbar(shrink=0.5)

plt.tight_layout()
# сохранить вместо show
plt.savefig("specaugment_debug.png", bbox_inches="tight", dpi=150)
print("Сохранён визуальный вывод: specaugment_debug.png")
