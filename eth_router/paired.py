"""Fit the two position decisions against the same historical price move."""
from copy import deepcopy

import numpy as np
import torch
from torch.nn import functional as F

from .market_env import settle


def fit_paired(router, features, scores, close, *, epochs, seed, device):
    """Batch Double Q updates with observed rewards for both possible actions."""
    def values(network, rows):
        rows = rows.cpu().numpy()
        # Adjacent rows represent flat and long starting positions at the same minute.
        x = [torch.as_tensor(a[rows], device=device).repeat_interleave(2, dim=0) for a in features]
        prior = torch.arange(2, device=device).repeat(len(rows))
        experts = torch.as_tensor(scores[rows], device=device).flatten(0, 1)
        return network(*x, prior, experts)[0].reshape(-1, 2, 2)

    yield from fit_action_values(router, values, close, epochs=epochs, seed=seed, device=device)


def fit_action_values(router, values, close, *, epochs, seed, device, discount=0.99,
                      batch_size=512, reward_bps=None, sampled=False, margin_weight=1.0):
    if sampled and margin_weight:
        raise ValueError("A sampled transition has no paired action margin")
    rng = np.random.default_rng(seed)
    action_rng = np.random.default_rng(seed + 1)
    optimizer = torch.optim.Adam(router.parameters(), lr=1e-4)
    target = deepcopy(router).requires_grad_(False)
    minutes = len(close) - 1
    positions = np.arange(2)
    transition = (close[:-1, None, None], close[1:, None, None],
                  positions[None, :, None], positions[None, None, :])
    rewards = settle(*transition)[0] if reward_bps is None else reward_bps(*transition)
    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)

    for epoch in range(epochs):
        errors = torch.zeros(2, device=device)
        order = rng.permutation(minutes)
        updates = 0
        for start in range(0, minutes, batch_size):
            rows = order[start:start + batch_size]
            rows = torch.as_tensor(rows, device=device)
            current = values(router, rows)
            with torch.no_grad():
                next_action = values(router, rows + 1).argmax(dim=-1, keepdim=True)
                continuation = values(target, rows + 1).gather(-1, next_action).squeeze(-1)
                continuation[rows == minutes - 1] = 0
                immediate = rewards[rows]
                # Today's target position becomes tomorrow's starting position.
                expected = immediate + discount * continuation[:, None, :]
            if sampled:
                # Use a separate stream so both arms see the same minibatch of minutes.
                combination = torch.as_tensor(action_rng.integers(4, size=(len(rows), 1)), device=device)
                selected = current.flatten(1).gather(1, combination)
                selected_target = expected.flatten(1).gather(1, combination)
                td = F.smooth_l1_loss(selected, selected_target)
            else:
                td = F.smooth_l1_loss(current, expected)
            margin = (F.smooth_l1_loss(current.diff(dim=-1), expected.diff(dim=-1))
                      if margin_weight else td.new_zeros(()))
            optimizer.zero_grad()
            (td + margin_weight * margin).backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                for lagged, online in zip(target.parameters(), router.parameters(), strict=True):
                    lagged.lerp_(online, 0.005)
            errors += len(rows) * torch.stack((td.detach(), margin.detach()))
            updates += 1
        td_mean, margin_mean = (errors / minutes).tolist()
        yield epoch + 1, {"updates": updates, "td_loss": td_mean,
                          "margin_loss": margin_mean, "transitions": minutes * (1 if sampled else 4)}
