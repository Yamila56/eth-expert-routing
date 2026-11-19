"""Replay the router over a prepared split and save its minute-level decisions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .checkpoint_source import ExpertSource
from .market_env import settle
from .policy import CONTEXT, ExpertRouter


def observations(frame, experts):
    return (*experts.market_inputs(frame),
            frame[["slope_360", "vol_360", *CONTEXT]].to_numpy(dtype=np.float32, copy=True))


@torch.inference_mode()
def simulate(router, inputs, scores, frame, device):
    count = len(frame) - 1
    if count < 1:
        raise ValueError("Replay needs at least two consecutive observations")
    parts = []
    for start in range(0, count, 4096):
        end = min(count, start + 4096)
        x = tuple(torch.as_tensor(array[start:end], device=device) for array in inputs)
        alternatives = []
        for prior in (0, 1):
            previous = torch.full((end - start,), prior, device=device, dtype=torch.long)
            q = torch.as_tensor(scores[start:end, prior], device=device)
            values, weights, threshold, mixed = router(*x, previous, q)
            votes = (q[..., 1] - q[..., 0]).sign()
            alternatives.append(torch.cat((values, mixed, threshold[:, None], weights, votes), dim=1).cpu().numpy())
        parts.append(np.stack(alternatives, axis=1))
    evaluations = np.concatenate(parts)
    choices = evaluations[:, :, :2].argmax(axis=2)
    actions = np.empty(count, dtype=np.int64)
    previous = np.empty_like(actions)
    prior = 0
    for minute, row in enumerate(choices):
        previous[minute] = prior
        prior = actions[minute] = row[prior]
    details = evaluations[np.arange(count), previous]
    close = frame["close"].to_numpy(dtype=np.float64)
    rewards, fees, cash_flow = settle(close[:-1], close[1:], previous, actions)
    return {
        "timestamp": pd.to_datetime(frame["timestamp"].iloc[:-1]).to_numpy(dtype="datetime64[ns]"),
        "previous_action": previous, "action": actions,
        "action_q": details[:, :2], "mixed_q": details[:, 2:4], "threshold": details[:, 4],
        "weights": details[:, 5:11], "votes": details[:, 11:17],
        "evidence": inputs[2][:-1, 2:], "evidence_columns": CONTEXT,
        "fee": fees, "reward": rewards, "cash_flow": cash_flow,
    }


def summarize(trace):
    pnl = np.r_[0, np.cumsum(trace["reward"])]
    required = max(0.0, -np.cumsum(trace["cash_flow"]).min())
    return {
        "validation_return": float(pnl[-1] / required) if required else 0.0,
        "net_pnl_usdt": float(pnl[-1]),
        "max_drawdown_usdt": float((np.maximum.accumulate(pnl) - pnl).max()),
        "switches": int(np.count_nonzero(trace["action"] != trace["previous_action"])),
        "holds_after_threshold": int(np.count_nonzero(trace["mixed_q"].argmax(axis=1) != trace["action"])),
    }


def main():
    parser = argparse.ArgumentParser()
    ExpertSource.add_arguments(parser)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    experts = ExpertSource.from_args(args, device)
    frame = pd.read_feather(args.input)
    inputs = observations(frame, experts)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    router = ExpertRouter(inputs[0].shape[1], inputs[1].shape[1], state["scale"].item()).to(device)
    router.load_state_dict(state)
    trace = simulate(router.eval(), inputs, experts.scores(frame), frame, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **trace)
    print(json.dumps(summarize(trace), indent=2))


if __name__ == "__main__":
    main()
