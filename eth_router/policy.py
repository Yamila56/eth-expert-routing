"""Shared expert weighting and a state-dependent switch threshold."""
import torch
from torch import nn
from torch.nn import functional as F

EXPERTS = ("slope_1", "slope_2", "slope_3", "vol_1", "vol_2", "vol_3")
FIELDS = ("pnl", "turnover", "history", "affinity", "available")
CONTEXT = [f"{expert}_{field}" for expert in EXPERTS for field in FIELDS]


class ExpertRouter(nn.Module):
    def __init__(self, state_size, trend_size, scale):
        super().__init__()
        self.state = nn.Linear(state_size + trend_size, 32)
        self.action = nn.Embedding(2, 32)
        self.identity = nn.Embedding(len(EXPERTS), 8)
        self.score = nn.Sequential(nn.Linear(64 + 2 + 8 + len(FIELDS) + 2, 32),
                                   nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)
        self.threshold = nn.Sequential(nn.Linear(64 + 2 + 2, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.threshold[-1].weight)
        nn.init.constant_(self.threshold[-1].bias, -8)
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))

    def encode(self, state, trend, previous):
        return torch.cat((self.state(torch.cat((state, trend), dim=1)), self.action(previous)), dim=1)

    def forward(self, state, trend, context, previous, scores):
        shared = self.encode(state, trend, previous)
        market = context[:, :2]
        evidence = context[:, 2:].reshape(-1, len(EXPERTS), len(FIELDS))
        margin = scores[:, :, 1] - scores[:, :, 0]
        votes = margin.sign()
        identity = self.identity.weight.expand(len(state), -1, -1)
        inputs = torch.cat((torch.cat((shared, market), dim=1)[:, None].expand(-1, len(EXPERTS), -1),
                            identity, evidence, votes[..., None],
                            (margin / self.scale).clamp(-8, 8)[..., None]), dim=-1)
        weights = self.score(inputs).squeeze(-1).softmax(dim=1)
        mixed = torch.bmm(weights[:, None], scores).squeeze(1)
        disagreement = 1 - votes.mean(dim=1).abs()
        affinity = (weights * evidence[:, :, 3] * evidence[:, :, 4]).sum(dim=1)
        gate_input = torch.cat((shared, market, disagreement[:, None], affinity[:, None]), dim=1)
        gap = (mixed[:, 1] - mixed[:, 0]).abs().detach()
        threshold = self.threshold(gate_input).sigmoid() * (gap[:, None] + self.scale)
        q = mixed - threshold * (1 - F.one_hot(previous, 2))
        return q, weights, threshold[:, 0], mixed
