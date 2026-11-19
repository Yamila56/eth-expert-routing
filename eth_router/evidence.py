"""Prepare causal expert evidence for train and validation."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .checkpoint_source import ExpertSource
from .policy import CONTEXT, EXPERTS


MINUTES_PER_FOLD = 30 * 1440
HOUR = 60
HISTORY = 360


class MarketVAE(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(size, 24), nn.SiLU(), nn.Linear(24, 8))
        self.decoder = nn.Sequential(nn.Linear(4, 24), nn.SiLU(), nn.Linear(24, size))

    def forward(self, values, sample=False):
        mean, logvar = self.encoder(values).chunk(2, dim=1)
        logvar = logvar.clamp(-8, 6)
        latent = mean + torch.randn_like(mean) * (0.5 * logvar).exp() if sample else mean
        return self.decoder(latent), mean, logvar


def market_state(frame):
    close = frame["close"]
    returns = np.log(close).diff()
    return pd.DataFrame({
        "return_bps": np.log(close / close.shift(HOUR)) * 10000,
        "vol_bps": returns.rolling(HOUR).std() * 10000,
        "spread_bps": ((frame["ask1_price"] - frame["bid1_price"]) / close).rolling(HOUR).mean() * 10000,
        "log_volume": np.log1p(frame["volume"].clip(lower=0)).rolling(HOUR).mean(),
        "imbalance": frame["volume_imbalance"].rolling(HOUR).mean(),
    }).to_numpy(dtype=np.float32)


def expert_actions(scores):
    choices = scores.argmax(axis=-1)
    positions = np.empty((len(choices), len(EXPERTS)), dtype=np.int8)
    previous = np.zeros(len(EXPERTS), dtype=np.int8)
    experts = np.arange(len(EXPERTS))
    for minute, row in enumerate(choices):
        previous = row[previous, experts]
        positions[minute] = previous
    return positions


def ledger(close, action, fee):
    old = np.vstack((np.zeros_like(action[:1]), action[:-1]))
    switches = np.abs(action - old)
    # PnL is expressed per ETH committed by a full-sized position.
    settled = np.vstack((np.zeros_like(action[:1], dtype=float),
                         action[:-1] * np.diff(close)[:, None] - switches[:-1] * close[:-1, None] * fee))
    prior_switches = np.vstack((np.zeros_like(switches[:1]), switches[:-1]))
    pnl = pd.DataFrame(settled).rolling(HISTORY, min_periods=1).sum().to_numpy() / close[:, None] * 10000
    turnover = pd.DataFrame(prior_switches).rolling(HISTORY, min_periods=1).sum().to_numpy() / HISTORY
    history = np.minimum(np.arange(len(close)), HISTORY) / HISTORY
    return settled, pnl, turnover, np.broadcast_to(history[:, None], pnl.shape)


def hourly_labels(settled, close):
    starts = np.arange(HOUR, len(settled) - HOUR, HOUR)
    cumsum = np.cumsum(settled, axis=0)
    gains = (cumsum[starts + HOUR] - cumsum[starts]) / close[starts, None] * 10000
    return starts, gains - np.median(gains, axis=1, keepdims=True)


def fit_models(features, starts, labels, boundary, device, epochs):
    # At the boundary t even the label ending at t has fully settled.
    eligible = starts + HOUR <= boundary
    x = features[starts[eligible]]
    y = labels[eligible]
    models = []
    for expert in range(labels.shape[1]):
        winners = x[y[:, expert] > 0]
        if len(winners) < 30:
            models.append(None)
            continue
        pivot = int(len(winners) * 0.8)
        fit, calibration = winners[:pivot], winners[pivot:]
        center = torch.as_tensor(fit.mean(axis=0), device=device)
        scale = torch.as_tensor(fit.std(axis=0).clip(1e-4), device=device)
        fit = (torch.as_tensor(fit, device=device) - center) / scale
        cal = (torch.as_tensor(calibration, device=device) - center) / scale
        model = MarketVAE(fit.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        for _ in range(epochs):
            for ids in torch.randperm(len(fit), device=device).split(256):
                batch = fit[ids]
                reconstructed, mean, logvar = model(batch, sample=True)
                loss = F.mse_loss(reconstructed, batch) - 0.025 * (1 + logvar - mean.square() - logvar.exp()).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            vae_errors = torch.sort((model(cal)[0] - cal).square().mean(dim=1)).values
            center_errors = torch.sort(cal.square().mean(dim=1)).values
        models.append((model, center, scale, vae_errors, center_errors))
    return models


@torch.inference_mode()
def affinities(models, features, device):
    valid = np.isfinite(features).all(axis=1)
    values = torch.as_tensor(features[valid], device=device)
    shape = (len(features), len(models))
    affinity = np.full(shape, 0.5, dtype=np.float32)
    distance = affinity.copy()
    available = np.zeros(shape, dtype=np.float32)
    for expert, fitted in enumerate(models):
        if fitted is None:
            continue
        model, center, scale, vae_errors, center_errors = fitted
        data = (values - center) / scale
        for errors, output, calibration in (
            ((model(data)[0] - data).square().mean(dim=1), affinity, vae_errors),
            (data.square().mean(dim=1), distance, center_errors),
        ):
            output[valid, expert] = (1 - torch.searchsorted(calibration, errors) / len(calibration)).cpu().numpy()
        available[valid, expert] = 1
    return affinity, distance, available


def frame_with_evidence(frame, evidence, columns):
    values = np.stack(evidence, axis=-1).reshape(len(frame), -1)
    return frame.join(pd.DataFrame(values.astype(np.float32), columns=columns))


def diagnostic(starts, labels, scores, available, names):
    result = {}
    for i, name in enumerate(names):
        high = (scores[starts, i] >= 0.75) & (available[starts, i] > 0)
        low = (scores[starts, i] <= 0.25) & (available[starts, i] > 0)
        result[name] = {
            "high_n": int(high.sum()), "low_n": int(low.sum()),
            "high_mean_relative_bps": float(labels[high, i].mean()) if high.any() else None,
            "low_mean_relative_bps": float(labels[low, i].mean()) if low.any() else None,
        }
    return result


def save_frame(frame, path):
    tmp = path.with_suffix(".tmp")
    frame.reset_index(drop=True).to_feather(tmp)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--fold-minutes", type=int, default=MINUTES_PER_FOLD)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--affinity", choices=("center", "vae"), default="center")
    parser.add_argument("--apply-test", action="store_true",
                        help="Apply the frozen train models to an explicitly prepared test split")
    args = parser.parse_args()
    if args.fold_minutes <= HOUR or args.epochs < 1 or args.output.resolve() == args.source.resolve():
        parser.error("Use a separate output path, positive epochs and a fold longer than one hour")
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    experts = ExpertSource(args.upstream, args.device)
    commission = 0.0002
    timings = {}
    datasets = {}
    splits = ("test",) if args.apply_test else ("train", "val")
    for split in splits:
        started = perf_counter()
        frame = pd.read_feather(source / f"{split}.feather")
        times = pd.to_datetime(frame["timestamp"])
        if not times.diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all():
            raise ValueError(f"{split}: expected contiguous minute observations")
        close = frame["close"].to_numpy()
        settled, pnl, turnover, history = ledger(close, expert_actions(experts.scores(frame)),
                                                    commission)
        datasets[split] = (frame, market_state(frame), (pnl, turnover, history), hourly_labels(settled, close))
        timings[f"expert_{split}_seconds"] = perf_counter() - started
    if args.apply_test:
        metadata = json.loads((output / "preparation.json").read_text(encoding="utf-8"))
        if metadata["fee"] != commission:
            raise ValueError("Fee differs from frozen training configuration")
        stored = torch.load(output / "affinity_models.pt", map_location=experts.device, weights_only=True)
        test, features, base, _ = datasets["test"]
        models = []
        for item in stored:
            if item is None:
                models.append(None)
                continue
            state, center, scale, vae_errors, center_errors = item
            model = MarketVAE(features.shape[1]).to(experts.device)
            model.load_state_dict(state)
            models.append((model.eval(), center, scale, vae_errors, center_errors))
        vae, center, available = affinities(models, features, experts.device)
        chosen = center if metadata["affinity"] == "center" else vae
        save_frame(frame_with_evidence(test, (*base, chosen, available), CONTEXT), output / "test.feather")
        return
    train, features, base, (starts, labels) = datasets["train"]
    affinity = np.full((len(train), len(EXPERTS)), 0.5, dtype=np.float32)
    distance = affinity.copy()
    available = np.zeros_like(affinity)
    warmup = args.fold_minutes
    if len(train) <= warmup + 1:
        raise ValueError("Train split is too short for the designated warmup")
    diagnostics = []
    started = perf_counter()
    for begin in range(warmup, len(train), args.fold_minutes):
        end = min(begin + args.fold_minutes, len(train))
        models = fit_models(features, starts, labels, begin, experts.device, args.epochs)
        affinity[begin:end], distance[begin:end], available[begin:end] = affinities(models, features[begin:end], experts.device)
        within = (starts >= begin) & (starts + HOUR < end)
        diagnostics.append({"start": str(train.timestamp.iloc[begin]), "end": str(train.timestamp.iloc[end-1]),
                            "vae": diagnostic(starts[within], labels[within], affinity, available, EXPERTS),
                            "center": diagnostic(starts[within], labels[within], distance, available, EXPERTS)})
    models = fit_models(features, starts, labels, len(train) - 1, experts.device, args.epochs)
    val, val_features, val_base, _ = datasets["val"]
    val_affinity, val_distance, val_available = affinities(models, val_features, experts.device)
    timings["fit_and_score_seconds"] = perf_counter() - started
    chosen_train = distance if args.affinity == "center" else affinity
    chosen_val = val_distance if args.affinity == "center" else val_affinity
    save_frame(frame_with_evidence(train, (*base, chosen_train, available), CONTEXT).iloc[warmup:], output / "train.feather")
    save_frame(frame_with_evidence(val, (*val_base, chosen_val, val_available), CONTEXT), output / "val.feather")
    state = [None if item is None else (item[0].state_dict(), *item[1:]) for item in models]
    torch.save(state, output / "affinity_models.pt")
    metadata = {"device": str(experts.device), "warmup_rows": warmup, "seed": args.seed, "epochs": args.epochs,
                "fee": commission, "affinity": args.affinity, "train_rows": len(train) - warmup, "val_rows": len(val),
                "train_start": str(train.timestamp.iloc[warmup]), "validation_start": str(val.timestamp.iloc[0]),
                "timings_s": timings, "walk_forward": diagnostics}
    (output / "preparation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in metadata.items() if key != "walk_forward"}), flush=True)


if __name__ == "__main__":
    main()
