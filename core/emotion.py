
class EmotionalTrace:
    def __init__(self, decay=0.95):
        self.rolling = 0.0
        self.decay = decay

    def update(self, surprise):
        self.rolling = self.decay * self.rolling + (1 - self.decay) * surprise

    def risk_bias(self):
        return min(1.0, self.rolling / 50.0)
