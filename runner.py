from env import PigEnv
from agent import PigAgent, RandomAgent, HoldAt20Agent
from typing import Optional, List
import random


class Runner:
    def __init__(self, players: int, agents: List[PigAgent], seed: Optional[int] = None):
        self.env = PigEnv(players, seed)
        self.agents = agents
        self.payouts = [0] * players

    def play_round(self):
        order = list(range(len(self.agents)))
        random.shuffle(order)
        pos_agents = [self.agents[i] for i in order]

        obs, _ = self.env.reset()
        done = False

        while not done:
            current_player = obs['current_player']
            agent = pos_agents[current_player]
            action = agent.act(obs)
            obs, reward, done, _, _ = self.env.step(action)

        round_payouts = self.env.get_payouts()

        mapped = [0] * len(self.payouts)
        for pos, payout in enumerate(round_payouts):
            original_agent_idx = order[pos]
            mapped[original_agent_idx] = payout

        return mapped

    def competition(self, rounds):
        for _ in range(rounds):
            round_payouts = self.play_round()
            for i in range(len(self.payouts)):
                self.payouts[i] += round_payouts[i]

    def output(self):
        print("Final Payouts:")
        for i, payout in enumerate(self.payouts):
            print(f"Player {i + 1}: {payout}")


if __name__ == '__main__':
    agents = [RandomAgent(), RandomAgent()]
    runner = Runner(players=2, agents=agents)
    runner.competition(rounds=10000)
    runner.output()
