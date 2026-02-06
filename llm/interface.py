import torch
from .aria_model import AriaLLM

class AriaLLMWrapper:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = AriaLLM().to(device)
        self.hidden = None

    def encode(self, text: str):
        data = torch.tensor(
            [list(text.encode("utf-8", errors="ignore")[:256])],
            dtype=torch.long,
            device=self.device
        )
        return data

    def predict(self, state_text: str):
        x = self.encode(state_text)

        with torch.no_grad():
            out = self.model(x, self.hidden)
            self.hidden = out["hidden"]

        # Sample a command token (single-step for now)
        token = torch.argmax(out["command_logits"], dim=-1).item()
        command = bytes([token]).decode("utf-8", errors="ignore")

        return {
            "predicted_stdout_len": out["stdout_len"].item(),
            "predicted_exit_code": round(out["exit_code"].item()),
            "confidence": out["confidence"].item(),
            "proposed_command": command
        }
