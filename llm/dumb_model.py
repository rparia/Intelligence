
import random

class DumbModel:
    def predict(self, state):
        command = random.choice([
            "ls",
            "pwd",
            "whoami",
            "date",
            "echo hello",
            "uname -a"
        ])

        return {
            "predicted_stdout": "",
            "predicted_exit_code": 0,
            "confidence": 0.3,
            "proposed_command": command
        }
