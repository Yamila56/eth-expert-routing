"""Test exit timing with a fixed schedule of identical entry opportunities."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .market_env import FEE_RATE, POSITION_ETH
from .outcomes import load_market
from .signal_study import FEATURE_WINDOWS, SEEDS, fit_signals, predict


CAP = 120
ENTRY_BPS = FEE_RATE * 10_000


def episodes(close, predictions):
    # Reserve the full evaluation window even when a policy exits early.
    entries = []
    for i in np.flatnonzero(predictions[:-CAP].mean(axis=1) >= ENTRY_BPS):
        if not entries or i > entries[-1] + CAP:
            entries.append(i)
    entry = np.asarray(entries, dtype=np.int64)
    offsets = np.arange(1, CAP + 1)
    rows = entry[:, None] + offsets
    prices = close[rows]
    cost = close[entry, None]
    observed_return = (close[rows - 1] / cost - 1) * 10_000
    age = np.broadcast_to(offsets / CAP, rows.shape)
    state = np.concatenate((predictions[rows], observed_return[..., None], age[..., None]), axis=-1)
    next_price = close[np.minimum(rows + 1, len(close) - 1)]
    # Action 0 pays the exit fee and terminates. Action 1 collects the next interval.
    payoff = np.stack((-FEE_RATE * prices / cost * 10_000,
                       (next_price - prices) / cost * 10_000), axis=-1)
    pnl = POSITION_ETH * (prices - cost - FEE_RATE * (prices + cost))
    return entry, state, payoff, pnl


def first_exit(exit_mask):
    exit_mask = exit_mask.copy()
    exit_mask[:, -1] = True
    return exit_mask.argmax(axis=1)


def evaluate(pnl, offsets):
    trade_pnl = pnl[np.arange(len(pnl)), offsets]
    return {"net_usdt": float(trade_pnl.sum()), "mean_trade_usdt": float(trade_pnl.mean()),
            "win_rate": float((trade_pnl > 0).mean()), "entries": len(pnl),
            "mean_exit_minutes": float((offsets + 1).mean())}


def fit(model, state, payoff, *, seed, epochs, sampled):
    target = deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    device = state.device
    transitions = np.flatnonzero(np.arange(state.shape[0] * CAP) % CAP != CAP - 1)
    state = state.flatten(0, 1)
    payoff = payoff.flatten(0, 1)
    order_rng = np.random.default_rng(seed)
    action_rng = np.random.default_rng(seed + 1)

    for epoch in range(epochs):
        updates = 0
        order = order_rng.permutation(transitions)
        for start in range(0, len(order), 512):
            rows = torch.as_tensor(order[start:start + 512], device=device)
            current = model(state[rows])
            with torch.no_grad():
                after = rows + 1
                best = model(state[after]).argmax(dim=1, keepdim=True)
                continuation = target(state[after]).gather(1, best).squeeze(1)
                continuation = torch.where(after % CAP == CAP - 1, payoff[after, 0], continuation)
                expected = payoff[rows].clone()
                expected[:, 1] += continuation
            if sampled:
                selected = torch.as_tensor(action_rng.integers(2, size=(len(rows), 1)), device=device)
                current = current.gather(1, selected)
                expected = expected.gather(1, selected)
            loss = F.smooth_l1_loss(current, expected)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                for moved, online in zip(target.parameters(), model.parameters(), strict=True):
                    moved.lerp_(online, 0.005)
            updates += 1
        yield epoch + 1, updates, len(order) * (1 if sampled else 2)


def model_for(width, device):
    return nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 48), nn.SiLU(),
                         nn.Linear(48, 2)).to(device)


@torch.inference_mode()
def choose(model, features):
    return first_exit(model(features).argmax(dim=-1).cpu().numpy() == 0)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--device", default="cpu")
    cli.add_argument("--epochs", type=int, default=12)
    args = cli.parse_args()
    if args.epochs < 1:
        cli.error("Epoch count must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    _, prices, features, columns = load_market(args.source, "train", windows=FEATURE_WINDOWS)
    cut = len(prices) * 3 // 5
    signals = fit_signals(features[:cut], prices[:cut], columns)
    _, training, rewards, _ = episodes(prices[cut:], predict(features[cut:], signals))
    if not len(training):
        raise ValueError("No training entries exceed the fixed entry level")

    def load_split(split):
        times, close, features, _ = load_market(args.source, split, windows=FEATURE_WINDOWS)
        entries, state, _, pnl = episodes(close, predict(features, signals))
        if not len(entries):
            raise ValueError(f"{split} has no entries above the fixed entry level")
        return times, entries, state, pnl

    mean = training.mean(axis=(0, 1))
    scale = training.std(axis=(0, 1)).clip(min=1e-8)

    def tensor(values):
        return torch.as_tensor(np.clip((values - mean) / scale, -10, 10), dtype=torch.float32, device=device)

    def rule_offsets(state):
        result = {f"hold_{h}": np.full(len(state), h - 1) for h in (5, 30, 120)}
        result["signal_fade"] = first_exit(state[..., :3].mean(axis=-1) <= -ENTRY_BPS)
        return result

    _, _, val_state, val_pnl = load_split("val")
    validation_tensor = tensor(val_state)
    val_rules = {name: evaluate(val_pnl, exits) for name, exits in rule_offsets(val_state).items()}
    selected_rule = max(val_rules, key=lambda name: val_rules[name]["net_usdt"])
    report = {"training_entries": len(training), "selected_rule": selected_rule, "validation_rules": val_rules,
              "runs": {}}
    train_tensor = tensor(training)
    reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    for seed in SEEDS:
        report["runs"][str(seed)] = {}
        for label, sampled in (("paired_exit", False), ("sampled_exit", True)):
            torch.manual_seed(seed)
            model = model_for(train_tensor.shape[-1], device)
            checkpoint = args.output / f"{seed}-{label}.pt"
            best = -np.inf
            curve = []
            for epoch, updates, transitions in fit(model, train_tensor, reward_tensor, seed=seed,
                                                    epochs=args.epochs, sampled=sampled):
                stats = evaluate(val_pnl, choose(model, validation_tensor))
                curve.append({"epoch": epoch, "updates": updates, "transitions": transitions, **stats})
                if stats["net_usdt"] > best:
                    best = stats["net_usdt"]
                    selected_epoch = epoch
                    torch.save(model.state_dict(), checkpoint)
            report["runs"][str(seed)][label] = {"selected_epoch": selected_epoch, "curve": curve}

    times, entries, test_state, test_pnl = load_split("test")
    test_tensor = tensor(test_state)

    def test(label, offsets):
        np.savez_compressed(args.output / f"{label}-historical-test.npz", entry_time=times[entries],
                            exit_time=times[entries + offsets + 1], pnl=test_pnl[np.arange(len(entries)), offsets])
        return evaluate(test_pnl, offsets)

    report["historical_test_rules"] = {label: test(label, offsets)
                                       for label, offsets in rule_offsets(test_state).items()}
    for seed, policies in report["runs"].items():
        for label, result in policies.items():
            model = model_for(test_tensor.shape[-1], device).eval()
            model.load_state_dict(torch.load(args.output / f"{seed}-{label}.pt", map_location=device, weights_only=True))
            result["historical_test"] = test(f"{seed}-{label}", choose(model, test_tensor))
    report["elapsed_seconds"] = round(perf_counter() - started, 1)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")


if __name__ == "__main__":
    main()
