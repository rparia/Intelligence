import torch
from .loss import aria_loss

def train_step(model, optimizer, prediction, outcome):
    optimizer.zero_grad()

    loss = aria_loss(prediction, outcome)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()
