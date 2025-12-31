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
        # return a lightweight view: tuple for scores (cheaper than creating numpy array each step)
        return {
            'scores': tuple(self.scores),
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

        # When the episode terminates expose full payouts so trainers can update all agents
        if terminated:
            payouts = self.get_payouts()
            reward = payouts[acting_player]
            info = {"payouts": payouts, "acting_player": acting_player}
        else:
            reward = 0
            info = {}

        return self._get_obs(), reward, terminated, truncated, info

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
            elif self.turn_total + self.scores[self.current_player] + d1 + d2 == 100:
                self._snake_eyes()
                break

            self.reroll_count = 0
            self.turn_total += int(d1 + d2)
            break

    def get_payouts(self):
        # Compute zero-sum payouts where payout_p = n * score_p - sum(all_scores)
        payouts = [0] * self.PLAYERS
        total = sum(self.scores)
        for p in range(self.PLAYERS):
            payouts[p] = int(self.PLAYERS * self.scores[p] - total)
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
