import json
import os
import torch
import yaml
import matplotlib.pyplot as plt

def write_run_yaml(run_dir, run_info, filename="run.yaml"):
    path = os.path.join(run_dir, filename)
    with open(path, "w") as f:
        f.write(
            yaml.safe_dump(
                run_info,
                sort_keys=False,
                default_flow_style=False,
            )
        )
    return

def flush_losses(run_dir, training_loss_history, training_accuracy_history, validation_loss_history, validation_accuracy_history):
    with open(os.path.join(run_dir, "losses.jsonl"), "w") as f:
        for epoch in range(len(training_loss_history)):
            rec = {
                "epoch": epoch,
                "training_loss": training_loss_history[epoch],
                "training_accuracy": training_accuracy_history[epoch],
                "validation_loss": validation_loss_history[epoch],
                "validation_accuracy": validation_accuracy_history[epoch],
            }
            f.write(json.dumps(rec) + "\n")

def save_checkpoint(checkpoint_dir, step, model, optimizer):
    filename = f"step_{step:08d}.pt"
    path = os.path.join(checkpoint_dir, filename)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    return

def load_checkpoint(checkpoint_dir, step):
    filename = f"step_{step:08d}.pt"
    path = os.path.join(checkpoint_dir, filename)
    return torch.load(path)

def plot_losses(num_epochs, training_loss_history, validation_loss_history, training_accuracy_history, validation_accuracy_history):
    epochs = range(num_epochs)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))

    ax_loss.plot(epochs, training_loss_history, label="train")
    ax_loss.plot(epochs, validation_loss_history, label="val")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.legend()

    ax_acc.plot(epochs, training_accuracy_history, label="train")
    ax_acc.plot(epochs, validation_accuracy_history, label="val")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.legend()

    plt.tight_layout()
    # plt.savefig(os.path.join(args.run_dir, "history.png"), dpi=150)
    plt.show()
    return


def load_losses_jsonl(run_dir):
    """Load training history from losses.jsonl file."""
    losses_path = os.path.join(run_dir, "losses.jsonl")
    epochs, train_loss, val_loss, train_acc, val_acc = [], [], [], [], []

    with open(losses_path, "r") as f:
        for line in f:
            rec = json.loads(line)
            epochs.append(rec["epoch"])
            train_loss.append(rec["training_loss"])
            val_loss.append(rec["validation_loss"])
            train_acc.append(rec["training_accuracy"])
            val_acc.append(rec["validation_accuracy"])

    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
    }


def find_grokking_epoch(val_acc, threshold=0.95):
    """Find the epoch where validation accuracy first exceeds threshold."""
    for i, acc in enumerate(val_acc):
        if acc >= threshold:
            return i
    return None


def plot_grokking_curves(run_dir, save_path=None):
    """
    Plot grokking curves in Neel Nanda's style.

    Shows the classic grokking pattern:
    - Training loss/accuracy converging quickly
    - Validation accuracy suddenly jumping after many epochs
    - Log scale for loss to see both phases clearly
    """
    data = load_losses_jsonl(run_dir)
    epochs = data["epochs"]

    # Find grokking transition
    grok_epoch = find_grokking_epoch(data["val_acc"], threshold=0.95)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot (log scale)
    ax_loss = axes[0]
    ax_loss.semilogy(epochs, data["train_loss"], label="Train", color="#2ecc71", linewidth=1.5)
    ax_loss.semilogy(epochs, data["val_loss"], label="Validation", color="#e74c3c", linewidth=1.5)
    if grok_epoch:
        ax_loss.axvline(x=grok_epoch, color="#3498db", linestyle="--", alpha=0.7,
                       label=f"Grokking (~epoch {grok_epoch})")
    ax_loss.set_xlabel("Epoch", fontsize=12)
    ax_loss.set_ylabel("Loss (log scale)", fontsize=12)
    ax_loss.set_title("Loss Curves", fontsize=14)
    ax_loss.legend(loc="upper right")
    ax_loss.grid(True, alpha=0.3)

    # Accuracy plot
    ax_acc = axes[1]
    ax_acc.plot(epochs, data["train_acc"], label="Train", color="#2ecc71", linewidth=1.5)
    ax_acc.plot(epochs, data["val_acc"], label="Validation", color="#e74c3c", linewidth=1.5)
    if grok_epoch:
        ax_acc.axvline(x=grok_epoch, color="#3498db", linestyle="--", alpha=0.7,
                      label=f"Grokking (~epoch {grok_epoch})")
    ax_acc.set_xlabel("Epoch", fontsize=12)
    ax_acc.set_ylabel("Accuracy", fontsize=12)
    ax_acc.set_title("Accuracy Curves", fontsize=14)
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.legend(loc="lower right")
    ax_acc.grid(True, alpha=0.3)

    plt.suptitle("Grokking: Sudden Generalization in Modular Addition", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()

    # Print summary
    if grok_epoch:
        print(f"\n--- Grokking Summary ---")
        print(f"Grokking occurred at epoch ~{grok_epoch}")
        print(f"Train accuracy at grokking: {data['train_acc'][grok_epoch]:.3f}")
        print(f"Final train accuracy: {data['train_acc'][-1]:.3f}")
        print(f"Final validation accuracy: {data['val_acc'][-1]:.3f}")

    return data