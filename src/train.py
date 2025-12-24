import torch
import torch.nn as nn
import tqdm as tqdm

def train(model, data_partial, data_full, training_config): # data_partial is DataLoader([torch.Tensor([a, b, =, c])])
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config['lr'], weight_decay=training_config['weight_decay'])
    loss_function = nn.CrossEntropyLoss()
    training_loss_history = []
    training_accuracy_history = []
    validation_loss_history = []
    validation_accuracy_history = []
    for _ in tqdm(range(training_config['epochs'])):
        correct = 0
        total = 0
        all_losses = []
        for batch in data_partial:
            output = model(batch[:, :-1])
            loss = loss_function(output.view(-1, output.size(-1)), batch[:, -1].view(-1))
            correct += (output.argmax(dim=-1) == batch[:, -1]).sum().item()
            total += batch.size(0)
            all_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        training_loss_history.append(torch.tensor(all_losses).mean().float().item())
        training_accuracy_history.append((correct / total).float())
        validation_loss, validation_accuracy = test(model, data_full)
        validation_loss_history.append(validation_loss)
        validation_accuracy_history.append(validation_accuracy.float())
    return training_loss_history, training_accuracy_history, validation_loss_history, validation_accuracy_history

def test(model, data_full): 
    loss_function = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    all_losses = []
    for batch in data_full:
        output = model(batch[:, :-1])
        loss = loss_function(output.view(-1, output.size(-1)), batch[:, -1].view(-1))
        all_losses.append(loss.item())
        correct += (output.argmax(dim=-1) == batch[:, -1]).sum().item()
        total += batch.size(0)
    return torch.tensor(all_losses).mean().float().item(), correct / total