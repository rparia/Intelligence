
import random

def select_action(prediction, emotional_trace):
    risk = emotional_trace.risk_bias()

    # As surprise rises, chance of executing drops
    if random.random() < risk:
        return None  # choose inaction

    return prediction["proposed_command"]
