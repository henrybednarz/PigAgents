from typing import Optional
import numpy as np


class PigAgent:
    def __init__(self):
        pass

    def act(self, obs):
        return 0


class RandomAgent(PigAgent):
    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        return self.rng.integers(0, 2)


class HoldAt20Agent(PigAgent):
    def __init__(self):
        super().__init__()

    def act(self, obs):
        if obs['turn_total'] >= 20:
            return 1
        else:
            return 0