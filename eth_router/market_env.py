"""Minute-by-minute settlement and the training-only planning target."""
from dataclasses import dataclass

import numpy as np


POSITION_ETH = 0.2
FEE_RATE = 0.0002


def settle(close, next_close, previous, action, fee_rate=FEE_RATE):
    """Execute at close, then value the position at next_close."""
    units = POSITION_ETH * (action - previous)
    fee = np.abs(units) * close * fee_rate
    reward = POSITION_ETH * action * (next_close - close) - fee
    cash_flow = -units * close - fee
    return reward, fee, cash_flow


@dataclass(frozen=True)
class Observation:
    minute: int
    position: int


@dataclass(frozen=True)
class Transition:
    observation: Observation
    action: int
    reward: float
    terminal: bool


class TradingEpisode:
    def __init__(self, prices):
        self.prices = np.asarray(prices, dtype=np.float64)
        if len(self.prices) < 2 or not np.isfinite(self.prices).all() or (self.prices <= 0).any():
            raise ValueError("Trading requires at least two finite positive prices")
        self.observation = Observation(0, 0)

    def reset(self, position=0):
        self.observation = Observation(0, position)
        return self.observation

    def step(self, action):
        previous = self.observation
        t = previous.minute
        if t >= len(self.prices) - 1:
            raise RuntimeError("Reset the episode before taking another action")
        reward, _, _ = settle(self.prices[t], self.prices[t + 1], previous.position, action)
        self.observation = Observation(t + 1, action)
        return Transition(previous, action, float(reward), t + 1 == len(self.prices) - 1)


def hindsight_values(prices, discount, teacher_fee=0.001):
    """Backward dynamic programming over the training price path.

    Values are indexed by minute, previous position and action. They are
    supervision targets built from training prices, never policy observations.
    """
    prices = np.asarray(prices, dtype=np.float64)
    actions = np.arange(2)[None, None, :]
    prior = np.arange(2)[None, :, None]
    reward, _, _ = settle(prices[:-1, None, None], prices[1:, None, None], prior, actions, teacher_fee)
    q = np.zeros((len(prices), 2, 2), dtype=np.float64)
    for t in range(len(reward) - 1, -1, -1):
        q[t] = reward[t] + discount * np.max(q[t + 1], axis=1)
    return q.astype(np.float32)
