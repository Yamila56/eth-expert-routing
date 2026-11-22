"""Causal minute features and delayed expert decision evidence."""
from pathlib import Path

import numpy as np
import pandas as pd

from .market_env import POSITION_ETH, settle


HALF_LIVES = (5, 30, 120)
OUTCOME_MINUTES = 30
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
    if len(frame) <= warmup + OUTCOME_MINUTES:
        raise ValueError(f"{split}: insufficient rows for feature warm-up and outcome evaluation")
    features = pd.DataFrame(columns).iloc[warmup:].to_numpy(dtype=np.float32)
    return (timestamps.iloc[warmup:].to_numpy(dtype="datetime64[ns]"),
            frame.close.iloc[warmup:].to_numpy(dtype=np.float64), features, list(columns))


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


def decision_evidence(close, opinions, *, horizon=OUTCOME_MINUTES, history=HISTORY_MINUTES):
    """Return [minute, starting position, expert, field] with settled observations only."""
    n, _, count = opinions.shape
    previous, actions = position_path(opinions)
    pnl, _, _ = settle(close[:-1, None], close[1:, None], previous[:-1], actions[:-1])
    zero = np.zeros((1, count))
    settled = rolling_sum(np.concatenate((zero, pnl)), history)
    turnover = rolling_sum(np.concatenate((zero, previous[:-1] != actions[:-1])), history)
    account = np.empty((n, count, 2), dtype=np.float32)
    account[..., 0] = settled / (POSITION_ETH * close[:, None]) * 10_000
    account[..., 1] = turnover / history

    # Compare holding the chosen position and holding the other position for H minutes.
    positions = np.arange(2)
    rewards, _, _ = settle(close[:-horizon, None, None], close[horizon:, None, None],
                           positions[None, :, None], opinions[:-horizon])
    alternate, _, _ = settle(close[:-horizon, None, None], close[horizon:, None, None],
                             positions[None, :, None], 1 - opinions[:-horizon])
    difference = (rewards - alternate) / (POSITION_ETH * close[:-horizon, None, None]) * 10_000
    delayed = np.zeros((n, 2, count, 2), dtype=np.float32)
    delayed[horizon:, ..., 0] = difference
    delayed[horizon:, ..., 1] = (np.sign(difference) + 1) / 2  # Half-credit for ties.
    total = rolling_sum(delayed, history)
    samples = np.clip(np.arange(n) - horizon + 1, 0, history)
    averages = total / np.maximum(samples, 1)[:, None, None, None]

    evidence = np.empty((n, 2, count, 5), dtype=np.float32)
    evidence[..., :2] = account[:, None]
    evidence[..., 2:4] = averages
    evidence[..., 4] = samples[:, None, None]
    return evidence
