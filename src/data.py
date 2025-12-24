import torch
from torch.utils.data import DataLoader, TensorDataset

def full_mod_add_table(p):
    a = torch.arange(p).repeat_interleave(p) # 0, ..., 1, ...
    b = torch.arange(p).repeat(p) # 0, 1, ..., 0, 1, ...
    eq = torch.full_like(a, p)

    x = torch.stack([a, b, eq], dim=1).long()
    y = (a + b) % p
    return x, y.long()


def make_mod_add_loaders(p, train_frac, batch_size=256):
    x_full, y_full = full_mod_add_table(p)
    full_ds = TensorDataset(x_full, y_full)

    n = x_full.shape[0]
    n_train = int(round(train_frac * n))
    idx = torch.randperm(n)[:n_train]

    train_ds = TensorDataset(x_full[idx], y_full[idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader