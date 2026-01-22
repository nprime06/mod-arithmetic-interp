import torch
from torch.utils.data import DataLoader, TensorDataset

def full_mod_add_table(p):
    a = torch.arange(p).repeat_interleave(p) # 0, ..., 1, ...
    b = torch.arange(p).repeat(p) # 0, 1, ..., 0, 1, ...
    eq = torch.full_like(a, p)

    x = torch.stack([a, b, eq], dim=1).long()
    y = (a + b) % p
    return x, y.long()


def make_mod_add_loaders(p, train_frac, batch_size=256, seed=42):
    """
    Create train/val dataloaders for modular addition.

    Uses a fixed seed for reproducible train/val splits (matching Nanda's setup).
    """
    x_full, y_full = full_mod_add_table(p)

    n = x_full.shape[0]
    n_train = int(round(train_frac * n))

    # Fixed seed for reproducible split
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:]  # Proper complement - no overlap!

    train_ds = TensorDataset(x_full[train_idx], y_full[train_idx])
    val_ds = TensorDataset(x_full[val_idx], y_full[val_idx])

    # full-batch training/eval: one batch per epoch
    train_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False, drop_last=False)
    return train_loader, val_loader