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

def flush_losses(loss_path, loss_buffer):
    with open(loss_path, "a") as f:
        for rec in loss_buffer:
            f.write(json.dumps(rec) + "\n")
    loss_buffer.clear()
    return

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

def plot_losses(epochs, training_loss_history, validation_loss_history, training_accuracy_history, validation_accuracy_history):
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