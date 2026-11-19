"""Frozen, position-conditioned dueling Q networks."""
import torch
from torch import nn
from torch.nn import functional as F


class ExpertQNetwork(nn.Module):
    """Estimate two actions under instantaneous and trend observations."""

    def __init__(self, market_size, trend_size, width=64):
        super().__init__()
        self.market = nn.Linear(market_size, width)
        self.trend = nn.Linear(trend_size, width)
        self.position = nn.Embedding(2, width)
        self.condition = nn.Linear(width, 2 * width)
        self.level = self._head(width, 1)
        self.preference = self._head(width, 2)

    @staticmethod
    def _head(width, out_size):
        return nn.Sequential(nn.Linear(width, width * 4), nn.GELU(approximate="tanh"),
                             nn.Linear(width * 4, out_size))

    def forward(self, market, trend, previous):
        context = self.trend(trend) + self.position(previous)
        offset, scale = self.condition(F.silu(context)).chunk(2, dim=-1)
        market_code = self.market(market)
        normalized = F.layer_norm(market_code, (market_code.shape[-1],), eps=1e-6)
        conditioned = normalized * (1 + scale) + offset
        advantage = self.preference(conditioned)
        return self.level(conditioned) + advantage - advantage.mean(dim=-1, keepdim=True)
