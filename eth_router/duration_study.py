"""Compare contextual holding-period selection with a validation-selected fixed period."""
import argparse
import json
from pathlib import Path

import numpy as np

from .exit_study import episodes
from .market_env import FEE_RATE, POSITION_ETH
from .outcomes import load_market
from .signal_study import FEATURE_WINDOWS, fit_signals, predict


PERIODS = np.array([5, 30, 60, 120])
RIDGE = 0.1


def account(close, entries, durations):
    exits = entries + durations
    fee = FEE_RATE * POSITION_ETH * (close[entries] + close[exits])
    trade_pnl = POSITION_ETH * (close[exits] - close[entries]) - fee
    position_change = np.zeros(len(close))
    position_change[entries] = POSITION_ETH
    position_change[exits] = -POSITION_ETH
    position = position_change.cumsum()
    cash_pnl = np.r_[0.0, position[:-1] * np.diff(close)]
    cash_pnl[entries] -= FEE_RATE * POSITION_ETH * close[entries]
    cash_pnl[exits] -= FEE_RATE * POSITION_ETH * close[exits]
    equity = cash_pnl.cumsum()
    np.testing.assert_allclose(equity[-1], trade_pnl.sum(), atol=1e-8)
    return {
        "entries": len(entries),
        "net_pnl_usdt": float(equity[-1]),
        "fees_usdt": float(fee.sum()),
        "max_drawdown_usdt": float((np.maximum.accumulate(equity) - equity).max()),
        "mean_hold_minutes": float(durations.mean()),
        "duration_counts": {str(h): int((durations == h).sum()) for h in PERIODS},
    }, trade_pnl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def load(split):
        return load_market(args.source, split, windows=FEATURE_WINDOWS)

    train_time, train_close, train_features, columns = load("train")
    boundary = len(train_close) * 3 // 5
    signals = fit_signals(train_features[:boundary], train_close[:boundary], columns)
    context_columns = [i for i, name in enumerate(columns)
                       if name.startswith(("volatility_", "spread_", "imbalance_"))
                       and not name.endswith("_2")]

    def prepare(market):
        times, close, features, _ = market
        forecasts = predict(features, signals)
        entries, _, _, outcomes = episodes(close, forecasts)
        if not len(entries):
            raise ValueError("The fixed entry schedule contains no eligible events")
        context = np.column_stack((forecasts[entries], features[np.ix_(entries, context_columns)]))
        return times, close, entries, context, outcomes[:, PERIODS - 1]

    _, train_price, entries, training, outcomes = prepare(
        (train_time[boundary:], train_close[boundary:], train_features[boundary:], columns)
    )
    mean = training.mean(axis=0)
    scale = training.std(axis=0).clip(min=1e-8)

    def design(features):
        x = np.clip((features - mean) / scale, -10, 10)
        return np.column_stack((np.ones(len(x)), x))

    x = design(training)
    target = 10_000 * outcomes / (POSITION_ETH * train_price[entries, None])
    penalty = np.eye(x.shape[1]) * len(x) * RIDGE
    penalty[0, 0] = 0
    weights = np.linalg.solve(x.T @ x + penalty, x.T @ target)
    np.savez(args.output / "duration_model.npz", periods=PERIODS, mean=mean, scale=scale,
             weights=weights, context_columns=np.asarray([columns[i] for i in context_columns]))

    def evaluate(prepared, split):
        times, close, events, features, _ = prepared
        predictions = design(features) @ weights
        adaptive = PERIODS[predictions.argmax(axis=1)]
        variants = {f"fixed_{h}": np.full(len(events), h) for h in PERIODS}
        variants["adaptive"] = adaptive
        metrics = {}
        for name, durations in variants.items():
            metrics[name], pnl = account(close, events, durations)
            np.savez_compressed(args.output / f"{split}-{name}.npz", entry_time=times[events],
                                exit_time=times[events + durations], durations=durations, pnl=pnl)
        return metrics

    validation = evaluate(prepare(load("val")), "validation")
    selected = max((f"fixed_{h}" for h in PERIODS), key=lambda name: validation[name]["net_pnl_usdt"])
    historical = evaluate(prepare(load("test")), "historical-test")
    report = {"training_entries": len(training), "selected_fixed": selected,
              "validation": validation, "historical_test": historical}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")


if __name__ == "__main__":
    main()
