
from llm.dumb_model import DumbModel
from world.shell import execute
from core.reward import compute_surprise
from core.emotion import EmotionalTrace
from core.arbiter import select_action
from core.memory import log

model = DumbModel()
emotion = EmotionalTrace()

def step():
    state = {}  # shell world is implicit

    prediction = model.predict(state)

    action = select_action(prediction, emotion)
    if action is None:
        return

    outcome = execute(action)

    surprise = compute_surprise(prediction, outcome)
    emotion.update(surprise)

    log({
        "prediction": prediction,
        "action": action,
        "outcome": outcome,
        "surprise": surprise,
        "emotional_state": emotion.rolling
    })
