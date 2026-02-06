import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 256          # byte-level
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
MAX_CMD_LEN = 64


class AriaLLM(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)

        self.rnn = nn.GRU(
            EMBED_DIM,
            HIDDEN_DIM,
            NUM_LAYERS,
            batch_first=True
        )

        # Prediction heads
        self.stdout_len_head = nn.Linear(HIDDEN_DIM, 1)
        self.exit_code_head = nn.Linear(HIDDEN_DIM, 1)
        self.confidence_head = nn.Linear(HIDDEN_DIM, 1)

        # Command generation head
        self.command_head = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, x, hidden=None):
        # x: (B, T) bytes
        emb = self.embedding(x)
        out, hidden = self.rnn(emb, hidden)

        h = out[:, -1]  # last timestep

        stdout_len = self.stdout_len_head(h)
        exit_code = self.exit_code_head(h)
        confidence = torch.sigmoid(self.confidence_head(h))

        command_logits = self.command_head(h)

        return {
            "stdout_len": stdout_len.squeeze(-1),
            "exit_code": exit_code.squeeze(-1),
            "confidence": confidence.squeeze(-1),
            "command_logits": command_logits,
            "hidden": hidden
        }
