"""Causal minute features and delayed expert decision evidence."""
from pathlib import Path

import numpy as np
import pandas as pd

from .market_env import FEE_RATE


HALF_LIVES = (5, 30, 120)
OUTCOME_WINDOWS = (5, 30, 120)
ACCOUNT_FIELDS = 2
OUTCOME_FIELDS = 3
HISTORY_MINUTES = 360
RAW_COLUMNS = ("timestamp", "close", "volume", "ask1_price", "bid1_price", "volume_imbalance")


def load_market(directory: Path, split: str, limit: int | None = None):
    frame = pd.read_feather(directory / f"df_{split}.feather", columns=list(RAW_COLUMNS)).iloc[:limit]
    timestamps = pd.to_datetime(frame["timestamp"])
    if not timestamps.diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all():
        raise ValueError(f"{split}: expected consecutive minute bars")
    if not np.isfinite(frame[list(RAW_COLUMNS[1:])]).all().all() or (frame.close <= 0).any():
        raise ValueError(f"{split}: invalid price, volume or order book observations")
    log_price = np.log(frame.close)
    returns = log_price.diff()
    volume = np.log1p(frame.volume.clip(lower=0))
    spread = (frame.ask1_price - frame.bid1_price) / frame.close
    columns = {}
    for window in HALF_LIVES:
        columns[f"return_{window}"] = log_price.diff(window)
        columns[f"volatility_{window}"] = returns.rolling(window).std()
        columns[f"volume_{window}"] = volume - volume.rolling(window).mean()
        columns[f"spread_{window}"] = spread.rolling(window).mean()
        columns[f"imbalance_{window}"] = frame.volume_imbalance.rolling(window).mean()
    warmup = max(HALF_LIVES)
    if len(frame) <= warmup + 2:
        raise ValueError(f"{split}: insufficient rows for feature warm-up and delayed execution")
    # Index every row by its simulated execution bar. Its observations stop one bar before.
    features = pd.DataFrame(columns).iloc[warmup:-1].to_numpy(dtype=np.float32)
    return (timestamps.iloc[warmup + 1:].to_numpy(dtype="datetime64[ns]"),
            frame.close.iloc[warmup + 1:].to_numpy(dtype=np.float64), features, list(columns))


def position_path(actions):
    """Select the candidate action for the position reached on each path."""
    n, _, count = actions.shape
    previous = np.zeros((n, count), dtype=np.intp)
    final = np.empty_like(previous)
    indices = np.arange(count)
    for minute, candidates in enumerate(actions):
        final[minute] = candidates[previous[minute], indices]
        if minute + 1 < n:
            previous[minute + 1] = final[minute]
    return previous, final


def rolling_sum(values, width):
    cumulative = np.cumsum(values, axis=0, dtype=np.float64)
    trailing = cumulative.copy()
    trailing[width:] -= cumulative[:-width]
    return trailing


def net_reward_bps(entry, mark, previous, action):
    return 10_000 * (action * (mark - entry) / entry - FEE_RATE * np.abs(action - previous))


def decision_evidence(close, opinions, *, horizons=OUTCOME_WINDOWS, history=HISTORY_MINUTES):
    """Return evidence for decisions made one bar before the indexed execution close."""
    n, _, count = opinions.shape
    previous, actions = position_path(opinions)
    # The interval [j, j+1] has settled when deciding for execution bar j+2.
    ledger = np.zeros((n, count, ACCOUNT_FIELDS))
    ledger[2:, :, 0] = net_reward_bps(close[:-2, None], close[1:-1, None],
                                     previous[:-2], actions[:-2])
    ledger[2:, :, 1] = previous[:-2] != actions[:-2]
    account = rolling_sum(ledger, history)
    evidence = np.zeros((n, 2, count, ACCOUNT_FIELDS + OUTCOME_FIELDS * len(horizons)), dtype=np.float32)
    evidence[..., :ACCOUNT_FIELDS] = account[:, None]
    evidence[..., 1] /= history

    positions = np.arange(2)[None, :, None]
    for index, horizon in enumerate(horizons):
        delay = horizon + 1
        entry = close[:-delay, None, None]
        mark = close[horizon:-1, None, None]
        chosen = opinions[:-delay]
        advantage = net_reward_bps(entry, mark, positions, chosen)
        advantage -= net_reward_bps(entry, mark, positions, 1 - chosen)

        delayed = np.zeros((n, 2, count, 2), dtype=np.float32)
        delayed[delay:, ..., 0] = advantage
        delayed[delay:, ..., 1] = (np.sign(advantage) + 1) / 2
        samples = np.clip(np.arange(n) - delay + 1, 0, history)
        start = ACCOUNT_FIELDS + OUTCOME_FIELDS * index
        evidence[..., start:start + 2] = rolling_sum(delayed, history) / np.maximum(samples, 1)[:, None, None, None]
        evidence[..., start + 2] = samples[:, None, None]
    return evidence
