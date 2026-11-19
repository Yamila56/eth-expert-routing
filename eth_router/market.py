import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


WINDOW = 360


def prepare(source, split, dest, limit):
    frame = pd.read_feather(source)
    if limit:
        frame = frame.iloc[:limit + WINDOW].copy()
    if not frame["timestamp"].is_monotonic_increasing or frame["timestamp"].duplicated().any():
        raise ValueError(f"{split}: timestamps must be strictly increasing")
    if not np.isfinite(frame["close"]).all() or (frame["close"] <= 0).any():
        raise ValueError(f"{split}: invalid close price")

    # The original filter and least-squares slope are linear within each window.
    # Compute their combined weights once. The window ends at the current row.
    b, a = butter(1, 0.05, btype="low")
    time = np.arange(WINDOW) - (WINDOW - 1) / 2
    slope_weights = time @ filtfilt(b, a, np.eye(WINDOW), axis=0) / (time @ time)
    frame["slope_360"] = np.r_[np.full(WINDOW - 1, np.nan), np.correlate(frame["close"], slope_weights)]
    frame["return"] = frame["close"].pct_change().fillna(0)
    frame["vol_360"] = frame["return"].rolling(WINDOW).std()
    frame = frame.dropna().iloc[1:].reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError(f"{split}: fewer than two usable rows")
    output = dest / f"{split}.feather"
    tmp = output.with_suffix(".tmp")
    frame.to_feather(tmp)
    tmp.replace(output)
    print(f"{split}: {len(frame):,} rows, {frame.timestamp.iloc[0]} to {frame.timestamp.iloc[-1]}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--splits", choices=("train", "val", "test"), nargs="+", default=("train", "val"))
    args = parser.parse_args()
    if args.limit is not None and args.limit < 2:
        parser.error("--limit must be at least 2")
    args.output.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        prepare(args.source / f"df_{split}.feather", split, args.output, args.limit)
