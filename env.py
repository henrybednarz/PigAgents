import gymnasium as gym
import numpy as np
from typing import Optional


class PigEnv(gym.Env):
    def __init__(self, players: int, seed: Optional[int] = None):
        super(PigEnv, self).__init__()

        self.PLAYERS = players

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict({
            'scores': gym.spaces.MultiDiscrete([200] * players),
            'turn_total': gym.spaces.Discrete(200),
            'current_player': gym.spaces.Discrete(players),
            'redemption': gym.spaces.Discrete(2)
        })

        self.scores = [0] * players
        self.round = 0
        self.game_over = False
        self.current_player = 0
        self.reroll_count = 0
        self.turn_total = 0
        self.redemption = False
        self.redemption_end = 0
        self.winner = None

        self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.round = 0
        self.scores = [0] * self.PLAYERS
        self.turn_total = 0
        self.current_player = 0
        self.reroll_count = 0
        self.game_over = False
        self.redemption = False
        self.redemption_end = 0
        self.winner = None
        return self._get_obs(), {}

    def _get_obs(self):
        return {
            'scores': np.array(self.scores, dtype=np.int64),
            'turn_total': int(self.turn_total),
            'current_player': int(self.current_player),
            'redemption': bool(self.redemption)
        }

    def step(self, action: int):
        if action not in (0, 1):
            raise ValueError("Invalid action. Must be 0 (roll) or 1 (hold).")

        acting_player = self.current_player

        if action == 0:
            self._roll()
        else:
            self._advance_turn()

        terminated = bool(self.game_over)
        truncated = False
        reward = 0

        if terminated:
            total = sum(self.scores)
            n = self.PLAYERS
            reward = int(n * self.scores[acting_player] - total)

        return self._get_obs(), reward, terminated, truncated, {}

    def _roll(self):
        while True:
            d1 = self.np_random.integers(1, 7)
            d2 = self.np_random.integers(1, 7)

            if (d1 == 1 and d2 == 1) or self.reroll_count == 3:
                self._snake_eyes()
                break

            if d1 == d2:
                self.turn_total += int(d1 * 4)
                self.reroll_count += 1
                continue

            if d1 + d2 == 7:
                self._seven()
                break

            self.reroll_count = 0
            self.turn_total += int(d1 + d2)
            break

    def get_payouts(self):
        payouts = [0] * self.PLAYERS
        max_score = max(self.scores)
        multipliers = [2 if score == 0 else 1 for score in self.scores]

        for p in range(self.PLAYERS):
            if p != self.winner:
                payouts[p] = multipliers[p] * (self.scores[p] - max_score)

        if self.winner is not None:
            payouts[self.winner] = -sum(payouts[p] for p in range(self.PLAYERS) if p != self.winner)

        return payouts

    def _snake_eyes(self):
        self.scores[self.current_player] = 0
        self.turn_total = 0
        self._advance_turn()

    def _seven(self):
        self.turn_total = 0
        self._advance_turn()

    def _advance_turn(self):
        self.scores[self.current_player] += int(self.turn_total)
        self.reroll_count = 0
        self.turn_total = 0

        if self.scores[self.current_player] > 100 and not self.redemption:
            self.redemption = True
            self.redemption_end = self.current_player

        elif (self.redemption and
              self.current_player != self.redemption_end and
              self.scores[self.current_player] == max(self.scores) and self.scores.count(max(self.scores)) == 1):
            self.redemption_end = self.current_player

        elif self.redemption and self.current_player == self.redemption_end:
            self.game_over = True
            self.winner = int(self.redemption_end)

        self.round += 1
        self.current_player = (self.current_player + 1) % self.PLAYERS

