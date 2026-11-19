"""Fit the two position decisions against the same historical price move."""
from copy import deepcopy

import numpy as np
import torch
from torch.nn import functional as F

from .market_env import settle


def fit_paired(router, features, scores, close, *, epochs, seed, device):
    """Batch Double Q updates with observed rewards for both possible actions."""
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(router.parameters(), lr=1e-4)
    target = deepcopy(router).requires_grad_(False)
    minutes = len(close) - 1
    positions = np.arange(2)
    rewards, _, _ = settle(close[:-1, None, None], close[1:, None, None],
                           positions[None, :, None], positions[None, None, :])

    def values(network, rows):
        # Adjacent rows represent flat and long starting positions at the same minute.
        x = [torch.as_tensor(a[rows], device=device).repeat_interleave(2, dim=0) for a in features]
        prior = torch.arange(2, device=device).repeat(len(rows))
        experts = torch.as_tensor(scores[rows], device=device).flatten(0, 1)
        return network(*x, prior, experts)[0].reshape(-1, 2, 2)

    for epoch in range(epochs):
        errors = np.zeros(2)
        order = rng.permutation(minutes)
        updates = 0
        for start in range(0, minutes, 512):
            rows = order[start:start + 512]
            current = values(router, rows)
            with torch.no_grad():
                next_action = values(router, rows + 1).argmax(dim=-1, keepdim=True)
                continuation = values(target, rows + 1).gather(-1, next_action).squeeze(-1)
                continuation[rows == minutes - 1] = 0
                immediate = torch.as_tensor(rewards[rows], dtype=current.dtype, device=device)
                # Today's target position becomes tomorrow's starting position.
                expected = immediate + 0.99 * continuation[:, None, :]
            td = F.smooth_l1_loss(current, expected)
            margin = F.smooth_l1_loss(current.diff(dim=-1), expected.diff(dim=-1))
            optimizer.zero_grad()
            (td + margin).backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                for lagged, online in zip(target.parameters(), router.parameters(), strict=True):
                    lagged.lerp_(online, 0.005)
            errors += len(rows) * np.array([td.item(), margin.item()])
            updates += 1
        yield epoch + 1, {"updates": updates, "td_loss": errors[0] / minutes,
                          "margin_loss": errors[1] / minutes}
