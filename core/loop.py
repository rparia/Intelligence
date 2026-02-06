
from llm.interface import AriaLLMWrapper
from llm.train_step import train_step
import torch
from world.shell import execute
from core.reward import compute_surprise
from core.emotion import EmotionalTrace
from core.arbiter import select_action
from core.memory import log

llm = AriaLLMWrapper(device="cuda")
optimizer = torch.optim.Adam(llm.model.parameters(), lr=1e-4)
emotion = EmotionalTrace()

def step():
    state = {}  # shell world is implicit

    prediction = llm.predict(state_text)

    action = select_action(prediction, emotion)
    if action is None:
        return

    outcome = execute(prediction["proposed_command"])
    surprise = compute_surprise(prediction, outcome)
    emotion.update(surprise)

    
  train_step(
    llm.model,
    optimizer,
    {
        "stdout_len": torch.tensor(prediction["predicted_stdout_len"]),
        "exit_code": torch.tensor(prediction["predicted_exit_code"]),
        "confidence": torch.tensor(prediction["confidence"])
    },
    {
        "stdout_len": len(outcome["stdout"]),
        "exit_code": outcome["exit_code"]
    }
)

    log({
        "prediction": prediction,
        "action": action,
        "outcome": outcome,
        "surprise": surprise,
        "emotional_state": emotion.rolling
    })
