import torch
from tqdm import tqdm
import os


def stable_cross_entropy(logits, labels):
    """
    Numerically stable cross-entropy loss using float64.

    Matches Nanda's implementation to avoid precision issues
    when logits become very confident (large magnitudes).
    """
    logits = logits.to(torch.float64)
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
    return -correct_log_probs.mean().float()


def train(model, data_partial, data_full, training_config, checkpoint_dir=None, checkpoint_every=1000):
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.lr, weight_decay=training_config.weight_decay, betas=training_config.betas)
    loss_function = stable_cross_entropy
    training_loss_history = []
    training_accuracy_history = []
    validation_loss_history = []
    validation_accuracy_history = []

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in tqdm(range(training_config.epochs)):
        correct = 0
        total = 0
        all_losses = []
        for x, y in data_partial:
            output = model(x)[:, -1, :]
            loss = loss_function(output, y)
            correct += (output.argmax(dim=-1) == y).sum().item()
            total += y.size(0)
            all_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        training_loss_history.append(torch.tensor(all_losses).mean().float().item())
        training_accuracy_history.append(correct / total)
        validation_loss, validation_accuracy = test(model, data_full)
        validation_loss_history.append(validation_loss)
        validation_accuracy_history.append(validation_accuracy)

        # Save checkpoint for mechanistic interpretability analysis
        if checkpoint_dir is not None and (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1:06d}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': training_loss_history[-1],
                'val_loss': validation_loss_history[-1],
                'train_acc': training_accuracy_history[-1],
                'val_acc': validation_accuracy_history[-1],
            }, checkpoint_path)

    # Save final checkpoint
    if checkpoint_dir is not None:
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_final.pt")
        torch.save({
            'epoch': training_config.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': training_loss_history[-1],
            'val_loss': validation_loss_history[-1],
            'train_acc': training_accuracy_history[-1],
            'val_acc': validation_accuracy_history[-1],
        }, checkpoint_path)

    return training_loss_history, training_accuracy_history, validation_loss_history, validation_accuracy_history

def test(model, data_full):
    loss_function = stable_cross_entropy
    correct = 0
    total = 0
    all_losses = []
    with torch.no_grad():
        for x, y in data_full:
            output = model(x)[:, -1, :]
            loss = loss_function(output, y)
            all_losses.append(loss.item())
            correct += (output.argmax(dim=-1) == y).sum().item()
            total += y.size(0)
    return torch.tensor(all_losses).mean().float().item(), correct / total