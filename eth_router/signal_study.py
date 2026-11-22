"""Test horizon-specific signals and matched position policies."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .horizons import HorizonExpert, MarketSplit, scores_for, validation_td_error
from .market_env import FEE_RATE, settle
from .outcomes import load_market, net_reward_bps
from .paired import fit_action_values


FEATURE_WINDOWS = (2, 5, 30, 120)
HORIZONS = (5, 30, 120)
SPECIALIST_WINDOWS = ((2, 5), (5, 30), (30, 120))
BUFFERS_BPS = (0.0, 2.0, 5.0, 10.0, 20.0, float("inf"))
SEEDS = (12345, 42, 2026)


def hold_return(close, horizon):
    return (close[horizon:] / close[:-horizon] - 1) * 10_000


def fit_signals(features, close, columns):
    models = []
    for horizon, windows in zip(HORIZONS, SPECIALIST_WINDOWS, strict=True):
        selected = [i for i, name in enumerate(columns) if int(name.rsplit("_", 1)[1]) in windows]
        inputs = features[:-horizon, selected].astype(np.float64)
        target = hold_return(close, horizon)
        mean = inputs.mean(axis=0)
        scale = inputs.std(axis=0).clip(min=1e-8)
        design = np.clip((inputs - mean) / scale, -10, 10)
        # Ridge strength is fixed per observation. The intercept is unpenalized.
        centered = design - design.mean(axis=0)
        coefficients = np.linalg.solve(centered.T @ centered + len(design) * 0.01 * np.eye(len(selected)),
                                       centered.T @ (target - target.mean()))
        intercept = target.mean() - design.mean(axis=0) @ coefficients
        models.append((selected, mean, scale, coefficients, intercept))
    return models


def predict(features, models):
    forecasts = []
    for selected, mean, scale, coefficients, intercept in models:
        forecasts.append(np.clip((features[:, selected] - mean) / scale, -10, 10) @ coefficients + intercept)
    return np.column_stack(forecasts)


def diagnose(close, predictions, upper):
    diagnostics = {}
    for i, horizon in enumerate(HORIZONS):
        target = hold_return(close, horizon)
        score = predictions[:-horizon, i]
        selected = score >= upper[i]
        # Buy at entry and sell at mark, including each fee at its own price.
        net = target - 10_000 * FEE_RATE * (2 + target / 10_000)
        diagnostics[str(horizon)] = {
            "upper_tail_count": int(selected.sum()),
            "upper_tail_gross_bps": float(target[selected].mean()) if selected.any() else None,
            "upper_tail_net_bps": float(net[selected].mean()) if selected.any() else None,
            "all_windows_net_bps": float(net.mean()),
        }
    return diagnostics


def replay_choices(candidates, close):
    actions = np.empty(len(candidates), dtype=np.int8)
    position = 0
    for i, pair in enumerate(candidates):
        position = actions[i] = pair[position]
    previous = np.r_[0, actions[:-1]]
    reward, fee, _ = settle(close[:-1], close[1:], previous, actions)
    equity = np.r_[0, reward.cumsum()]
    results = {"gross_pnl_usdt": float(reward.sum() + fee.sum()), "net_pnl_usdt": float(equity[-1]),
               "fees_usdt": float(fee.sum()), "max_drawdown_usdt": float((np.maximum.accumulate(equity)-equity).max()),
               "switches": int((previous != actions).sum()), "minutes": len(actions)}
    return results, {"action": actions, "previous_action": previous, "reward": reward, "fee": fee}


def rule_choices(predictions, buffer):
    edge = predictions[:-1].mean(axis=1)
    band = 10_000 * FEE_RATE + buffer
    return np.column_stack((edge > band, edge >= -band)).astype(np.int8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", default=12, type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rule-only", action="store_true",
                        help="Run the forecast-driven rule and two economic references")
    args = parser.parse_args()
    if args.epochs < 1 or (args.limit is not None and args.limit < 512):
        parser.error("Use positive epochs and at least 512 raw rows for a smoke run")
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    torch.set_num_threads(1)
    device = torch.device(args.device)

    def read(split):
        return load_market(args.source, split, args.limit, windows=FEATURE_WINDOWS)

    train_time, train_close, train_raw, columns = read("train")
    boundary = len(train_close) * 3 // 5
    if min(boundary, len(train_close) - boundary) <= max(HORIZONS):
        parser.error("Both training stages require more than 120 available minutes")
    models = fit_signals(train_raw[:boundary], train_close[:boundary], columns)
    training_predictions = predict(train_raw[boundary:], models)
    mean = training_predictions.mean(axis=0)
    scale = training_predictions.std(axis=0).clip(min=1e-8)
    upper = np.quantile(training_predictions, 0.9, axis=0)

    def market(time, close, prediction):
        return MarketSplit(time, close, ((prediction - mean) / scale).clip(-10, 10).astype(np.float32), device)

    train = market(train_time[boundary:], train_close[boundary:], training_predictions)
    val_time, val_close, val_raw, _ = read("val")
    val_predictions = predict(val_raw, models)
    val = market(val_time, val_close, val_predictions)
    rule_validation = [replay_choices(rule_choices(val_predictions, buffer), val_close)[0]
                       for buffer in BUFFERS_BPS]
    chosen_rule = max(range(len(rule_validation)), key=lambda i: rule_validation[i]["net_pnl_usdt"])
    buffer = BUFFERS_BPS[chosen_rule]
    report = {
        "signal_diagnostics": {"validation": diagnose(val_close, val_predictions, upper)},
        "rule": {"buffer_bps": buffer if np.isfinite(buffer) else "always_flat",
                 "validation": rule_validation[chosen_rule]},
        "runs": {},
    }
    np.savez(args.output / "signal_scaling.npz", mean=mean, scale=scale, upper=upper)
    (args.output / "signal_models.json").write_text(json.dumps({
        "columns": columns, "models": [[np.asarray(x).tolist() for x in model] for model in models],
    }, indent=2), encoding="utf8")
    print({"rule": report["rule"], "signal_validation": report["signal_diagnostics"]["validation"]}, flush=True)

    seeds = () if args.rule_only else SEEDS
    for seed in seeds:
        report["runs"][str(seed)] = {}
        for label, sampled in (("paired_td", False), ("sampled_td", True)):
            torch.manual_seed(seed)
            policy = HorizonExpert(len(HORIZONS)).to(device)
            checkpoint = args.output / f"{seed}-{label}.pt"
            best = -np.inf
            curve = []
            for epoch, loss in fit_action_values(policy, train.values, train.close,
                                                 epochs=args.epochs, seed=seed, device=device,
                                                 reward_bps=net_reward_bps, sampled=sampled, margin_weight=0.0):
                q = scores_for(lambda rows: val.values(policy, rows), len(val_close), device)
                result, _ = replay_choices(q[:-1].argmax(axis=-1), val_close)
                curve.append({"epoch": epoch, **loss, **result,
                              "validation_bellman_huber": validation_td_error(q, val_close)})
                if result["net_pnl_usdt"] > best:
                    best = result["net_pnl_usdt"]
                    selected_epoch = epoch
                    torch.save(policy.state_dict(), checkpoint)
                print({"seed": seed, "policy": label, "epoch": epoch, **loss,
                       "validation_net": result["net_pnl_usdt"], "seconds": round(perf_counter()-started, 1)}, flush=True)
            report["runs"][str(seed)][label] = {"selected_epoch": selected_epoch, "curve": curve}

    # Read the historical test only after every validation choice is locked.
    test_time, test_close, test_raw, _ = read("test")
    test_predictions = predict(test_raw, models)
    test = market(test_time, test_close, test_predictions)
    report["signal_diagnostics"]["historical_test"] = diagnose(test_close, test_predictions, upper)

    def test_replay(label, candidates):
        result, trace = replay_choices(candidates, test_close)
        np.savez_compressed(args.output / f"{label}-historical-test.npz", **trace,
                            execution_timestamp=test_time[:-1])
        return result

    report["rule"]["historical_test"] = test_replay("rule", rule_choices(test_predictions, buffer))
    report["always_flat"] = test_replay("always-flat", np.zeros((len(test_close) - 1, 2), dtype=np.int8))
    report["buy_and_hold"] = test_replay("buy-and-hold", np.ones((len(test_close) - 1, 2), dtype=np.int8))
    for seed, runs in report["runs"].items():
        for label, run in runs.items():
            policy = HorizonExpert(len(HORIZONS)).to(device).eval()
            policy.load_state_dict(torch.load(args.output / f"{seed}-{label}.pt", map_location=device, weights_only=True))
            q = scores_for(lambda rows: test.values(policy, rows), len(test_close), device)
            run["historical_test"] = test_replay(f"{seed}-{label}", q[:-1].argmax(axis=-1))
    report["execution"] = {"device": str(device), "epochs": args.epochs, "limit": args.limit,
                           "fee_bps": 10_000 * FEE_RATE, "elapsed_seconds": round(perf_counter() - started, 1)}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")


if __name__ == "__main__":
    main()
