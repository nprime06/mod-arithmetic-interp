import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run-dir', type=str, required=True)
parser.add_argument('--p', type=int, required=True)
args = parser.parse_args()

import os
from dataclasses import dataclass, asdict
from model import ModAddModel
from data import make_mod_add_loaders
from train import train
from run_io import write_run_yaml, flush_losses, save_checkpoint, plot_losses

@dataclass
class ModelConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    h_q: int
    h_k: int
    d_model: int
    d_head: int

@dataclass
class TrainingConfig:
    p: int
    train_frac: float
    batch_size: int
    lr: float
    weight_decay: float
    epochs: int
    run_dir: str

model_config = ModelConfig(
    block_size=4,
    vocab_size=args.p+1,
    n_layer=1,
    h_q=4,
    h_k=4,
    d_model=128,
    d_head=32,
)

training_config = TrainingConfig(
    p=args.p,
    train_frac=0.7,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-2,
    epochs=100,
)

run_info = {
    "p": args.p,
    **asdict(model_config),
    **asdict(training_config),
}

write_run_yaml(args.run_dir, run_info)
data_partial, data_full = make_mod_add_loaders(args.p, training_config.train_frac, training_config.batch_size)
model = ModAddModel(model_config)
training_loss_history, training_accuracy_history, validation_loss_history, validation_accuracy_history = train(model, data_partial, data_full, training_config)

plot_losses(training_config.epochs, training_loss_history, validation_loss_history, training_accuracy_history, validation_accuracy_history)
