import torch
import torch.nn.functional as F

def aria_loss(pred, outcome):
    # Prediction error
    stdout_loss = F.mse_loss(
        pred["stdout_len"],
        torch.tensor(outcome["stdout_len"], device=pred["stdout_len"].device)
    )

    exit_loss = F.mse_loss(
        pred["exit_code"],
        torch.tensor(outcome["exit_code"], device=pred["exit_code"].device)
    )

    # Confidence penalty
    error = stdout_loss + exit_loss
    confidence_penalty = pred["confidence"] * error.detach()

    return stdout_loss + exit_loss + confidence_penalty
