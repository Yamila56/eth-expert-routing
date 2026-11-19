"""Train the router with an online account path and batched Double DQN updates."""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .checkpoint_source import ExpertSource
from .policy import ExpertRouter
from .replay import FEE_RATE, POSITION_ETH, observations, simulate, summarize


def fit(router, features, scores, close, *, epochs, seed, device):
    rng = np.random.default_rng(seed)
    target = deepcopy(router).requires_grad_(False)
    optimizer = torch.optim.Adam(router.parameters(), lr=0.0001)
    capacity = min(1_000_000, epochs * (len(close) - 1))
    minutes = np.empty(capacity, dtype=np.int64)
    priors = np.empty(capacity, dtype=np.int64)
    actions = np.empty(capacity, dtype=np.int64)
    rewards = np.empty(capacity, dtype=np.float32)
    steps = 0
    switches = np.array([[0, 1], [1, 0]])

    def values(model, t, previous):
        market = [torch.as_tensor(array[t], device=device) for array in features]
        q = torch.as_tensor(scores[t, previous], device=device)
        return model(*market, torch.as_tensor(previous, device=device), q)[0]

    for epoch in range(epochs):
        previous = 0
        epsilon = max(0.30, 0.70 - epoch * 0.08)
        updates = 0
        for t in range(len(close) - 1):
            with torch.no_grad():
                action = (int(rng.integers(2)) if rng.random() < epsilon else
                          values(router, np.array([t]), np.array([previous])).argmax().item())
            fee = switches[previous, action] * close[t] * POSITION_ETH * FEE_RATE
            reward = action * POSITION_ETH * (close[t + 1] - close[t]) - fee
            slot = steps % capacity
            minutes[slot], priors[slot], actions[slot], rewards[slot] = t, previous, action, reward
            previous = action
            steps += 1
            if steps <= 512 or steps % 512:
                continue
            for _ in range(10):
                sample = rng.integers(min(steps, capacity), size=512)
                placed, at = actions[sample], minutes[sample]
                ar = torch.as_tensor(placed, device=device)
                predictions = values(router, at, priors[sample]).gather(1, ar[:, None]).squeeze(1)
                with torch.no_grad():
                    candidate = values(router, at + 1, placed).argmax(dim=1, keepdim=True)
                    future = values(target, at + 1, placed).gather(1, candidate).squeeze(1)
                    active = torch.as_tensor(at + 1 < len(close) - 1, device=device)
                    expected = torch.as_tensor(rewards[sample], device=device) + 0.99 * active * future
                loss = F.mse_loss(predictions, expected)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(router.parameters(), 1)
                optimizer.step()
                with torch.no_grad():
                    for dst, src in zip(target.parameters(), router.parameters(), strict=True):
                        dst.lerp_(src, 0.005)
                updates += 1
        yield epoch + 1, updates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-device", default="cpu")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    experts = ExpertSource(args.upstream, args.score_device)
    device = torch.device(args.device)
    train = pd.read_feather(args.data / "train.feather")
    val = pd.read_feather(args.data / "val.feather")
    if len(train) <= 1024 or len(val) < 2:
        parser.error("Provide at least 1025 training observations and two validation observations")
    train_inputs, val_inputs = observations(train, experts), observations(val, experts)
    train_scores, val_scores = experts.scores(train), experts.scores(val)
    router = ExpertRouter(train_inputs[0].shape[1], train_inputs[1].shape[1], 0.05).to(device)
    args.output.mkdir(parents=True, exist_ok=True)
    best = -float("inf")
    for epoch, updates in fit(router, train_inputs, train_scores, train["close"].to_numpy(),
                              epochs=args.epochs, seed=args.seed, device=device):
        trace = simulate(router, val_inputs, val_scores, val, device)
        metrics = summarize(trace)
        print({"epoch": epoch, "updates": updates, **metrics}, flush=True)
        if metrics["net_pnl_usdt"] > best:
            best = metrics["net_pnl_usdt"]
            torch.save(router.state_dict(), args.output / "router.pt")
            np.savez_compressed(args.output / "expert_trace.npz", **trace)


if __name__ == "__main__":
    main()
