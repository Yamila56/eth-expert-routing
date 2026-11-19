"""Train the router with an online account path and batched Double DQN updates."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .agent import RoutingAgent
from .checkpoint_source import ExpertSource
from .market_env import TradingEpisode, hindsight_values
from .policy import ExpertRouter
from .replay import observations, simulate, summarize


def fit(router, features, scores, close, *, epochs, seed, device):
    episode = TradingEpisode(close)
    capacity = min(1_000_000, epochs * (len(close) - 1))
    teacher = hindsight_values(close, discount=0.99)
    agent = RoutingAgent(router, features, scores, teacher, capacity, seed, device)
    for epoch in range(epochs):
        state = episode.reset(position=int(agent.random.integers(2)))
        epsilon = max(0.30, 0.70 - epoch * 0.08)
        updates = 0
        losses = np.zeros(3)
        for _ in range(len(close) - 1):
            transition = episode.step(agent.act(state, epsilon))
            agent.observe(transition)
            state = episode.observation
            steps = agent.replay.total
            if steps <= 513 or steps % 512:
                continue
            for _ in range(10):
                losses += agent.learn(512)
                updates += 1
            if steps > agent.memory.capacity:
                agent.memory.refresh_keys(agent.encode)
        td, memory, demo = losses / max(1, updates)
        yield epoch + 1, {
            "updates": updates, "td_loss": float(td), "memory_loss": float(memory),
            "demonstration_kl": float(demo), "memory_hits_total": agent.memory.hits,
            "memory_refreshes_total": agent.memory.refreshes,
        }


def main():
    parser = argparse.ArgumentParser()
    ExpertSource.add_arguments(parser)
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
    experts = ExpertSource.from_args(args, args.score_device)
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
    for epoch, learning in fit(router, train_inputs, train_scores, train["close"].to_numpy(),
                              epochs=args.epochs, seed=args.seed, device=device):
        trace = simulate(router, val_inputs, val_scores, val, device)
        metrics = summarize(trace)
        print({"epoch": epoch, **learning, **metrics}, flush=True)
        if metrics["validation_return"] > best:
            best = metrics["validation_return"]
            torch.save(router.state_dict(), args.output / "router.pt")
            np.savez_compressed(args.output / "expert_trace.npz", **trace)


if __name__ == "__main__":
    main()
