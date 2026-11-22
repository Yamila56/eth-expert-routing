"""Train and evaluate three horizon experts and an independent position policy."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from .market_env import settle
from .outcomes import HALF_LIVES, HISTORY_MINUTES, decision_evidence, load_market, position_path
from .paired import fit_action_values


class HorizonExpert(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.values = nn.Sequential(nn.Linear(features + 1, 48), nn.SiLU(),
                                    nn.Linear(48, 48), nn.SiLU(), nn.Linear(48, 2))

    def forward(self, market, position):
        return self.values(torch.cat((market, position), dim=-1))


class PositionPolicy(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.expert_encoder = nn.Sequential(nn.Linear(len(HALF_LIVES) + 2 + 5, 32), nn.SiLU(), nn.Linear(32, 32))
        self.values = nn.Sequential(nn.Linear(features + 33, 48), nn.SiLU(), nn.Linear(48, 2))

    def forward(self, market, position, evidence):
        encoded = self.expert_encoder(evidence).mean(dim=-2)
        return self.values(torch.cat((market, position, encoded), dim=-1))


class MarketSplit:
    def __init__(self, time, close, features, device):
        self.time = time
        self.close = close
        market = torch.as_tensor(features, device=device)
        self.market = market[:, None, :].expand(-1, 2, -1)
        positions = torch.arange(2, dtype=torch.float32, device=device)
        self.position = positions[None, :, None].expand(len(close), -1, -1)

    def values(self, network, rows):
        return network(self.market[rows], self.position[rows])


@torch.inference_mode()
def scores_for(values, length, device):
    rows = torch.arange(length, device=device)
    return np.concatenate([values(batch).cpu().numpy() for batch in rows.split(8192)])


def expert_scores(experts, market, device):
    return np.stack([scores_for(lambda rows: market.values(expert, rows),
                                len(market.close), device) for expert in experts], axis=2)


def policy_inputs(opinions, margins, evidence, device, *, include_outcomes):
    shape = opinions.shape
    identity = np.broadcast_to(np.eye(len(HALF_LIVES), dtype=np.float32), (*shape, len(HALF_LIVES)))
    # Use fixed units for the ledger and sample coverage. Keep raw fields for the replay.
    records = evidence.copy()
    records[..., (0, 2)] /= 100
    records[..., 4] /= HISTORY_MINUTES
    if not include_outcomes:
        records[..., 2:] = 0
    result = np.concatenate((identity, opinions[..., None], margins[..., None], records), axis=-1, dtype=np.float32)
    return torch.as_tensor(result, device=device)


def replay(values, market, opinions, margins, evidence):
    previous, action = position_path(values[:-1].argmax(axis=-1)[..., None])
    previous, action = previous[:, 0], action[:, 0]
    reward, fee, _ = settle(market.close[:-1], market.close[1:], previous, action)
    row = np.arange(len(action))
    return {"timestamp": market.time[:-1], "action": action, "previous_action": previous,
            "action_q": values[row, previous], "votes": opinions[row, previous],
            "margins": margins[row, previous], "evidence": evidence[row, previous],
            "reward": reward, "fee": fee, "half_lives": np.asarray(HALF_LIVES)}


def metrics(trace):
    equity = np.r_[0, trace["reward"].cumsum()]
    votes = trace["votes"]
    return {"minutes": len(votes), "net_pnl_usdt": float(equity[-1]),
            "max_drawdown_usdt": float(np.max(np.maximum.accumulate(equity) - equity)),
            "fees_usdt": float(trace["fee"].sum()),
            "switches": int(np.count_nonzero(trace["action"] != trace["previous_action"])),
            "disagreement_minutes": int(np.count_nonzero((votes != votes[:, :1]).any(axis=1)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, help="Use the first N raw rows per split for a smoke run")
    parser.add_argument("--expert-epochs", type=int, default=12)
    parser.add_argument("--policy-epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--test", action="store_true", help="Evaluate both frozen policies on the test split")
    args = parser.parse_args()
    if min(args.expert_epochs, args.policy_epochs) < 1 or (args.limit is not None and args.limit < 300):
        parser.error("Use positive epoch counts and at least 300 raw rows per split")
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    time, close, raw, columns = load_market(args.source, "train", args.limit)
    boundary = len(raw) * 3 // 5
    mean = raw[:boundary].mean(axis=0)
    scale = raw[:boundary].std(axis=0).clip(min=1e-8)
    features = ((raw - mean) / scale).clip(-10, 10)
    expert_train = MarketSplit(time[:boundary], close[:boundary], features[:boundary], device)
    policy_train = MarketSplit(time[boundary:], close[boundary:], features[boundary:], device)

    experts = []
    for half_life in HALF_LIVES:
        torch.manual_seed(args.seed)
        expert = HorizonExpert(len(columns)).to(device)
        for epoch, loss in fit_action_values(expert, expert_train.values, expert_train.close,
                                             epochs=args.expert_epochs, seed=args.seed, device=device,
                                             discount=2 ** (-1 / half_life)):
            print({"expert_half_life": half_life, "epoch": epoch, **loss,
                   "elapsed_seconds": round(perf_counter() - started, 1)}, flush=True)
        experts.append(expert.eval().requires_grad_(False))
    q = expert_scores(experts, expert_train, device)
    margin_scale = np.maximum(np.sqrt(np.mean(np.square(q[..., 1] - q[..., 0]), axis=(0, 1))), 1e-6)
    torch.save({"states": [expert.state_dict() for expert in experts],
                "columns": columns, "feature_mean": torch.from_numpy(mean),
                "feature_scale": torch.from_numpy(scale), "margin_scale": torch.from_numpy(margin_scale),
                "half_lives": HALF_LIVES, "seed": args.seed}, args.output / "experts.pt")

    def load_split(split):
        time, close, raw, _ = load_market(args.source, split, args.limit)
        return MarketSplit(time, close, ((raw - mean) / scale).clip(-10, 10), device)

    def prepare(market):
        q = expert_scores(experts, market, device)
        opinions = q.argmax(axis=-1)
        margins = ((q[..., 1] - q[..., 0]) / margin_scale).clip(-8, 8)
        return opinions, margins, decision_evidence(market.close, opinions)

    def evaluator(market, data, include_outcomes):
        inputs = policy_inputs(*data, device, include_outcomes=include_outcomes)

        def values(network, rows):
            return network(market.market[rows], market.position[rows], inputs[rows])

        return values

    validation = load_split("val")
    training_data = prepare(policy_train)
    validation_data = prepare(validation)
    results = {}
    for label, include_outcomes in (("full", True), ("ledger_only", False)):
        torch.manual_seed(args.seed)
        policy = PositionPolicy(len(columns)).to(device)
        training_values = evaluator(policy_train, training_data, include_outcomes)
        validation_values = evaluator(validation, validation_data, include_outcomes)
        best = -np.inf
        for epoch, loss in fit_action_values(policy, training_values, policy_train.close,
                                             epochs=args.policy_epochs, seed=args.seed, device=device):
            values = scores_for(lambda rows: validation_values(policy, rows), len(validation.close), device)
            trace = replay(values, validation, *validation_data)
            summary = metrics(trace)
            print({"policy": label, "epoch": epoch, **loss, **summary,
                   "elapsed_seconds": round(perf_counter() - started, 1)}, flush=True)
            if summary["net_pnl_usdt"] > best:
                best = summary["net_pnl_usdt"]
                results[label] = {"validation": summary, "selected_epoch": epoch}
                torch.save(policy.state_dict(), args.output / f"{label}.pt")
                np.savez_compressed(args.output / f"{label}_val.npz", **trace)

    if args.test:
        test = load_split("test")
        test_data = prepare(test)
        traces = {}
        for label in results:
            policy = PositionPolicy(len(columns)).to(device).eval()
            policy.load_state_dict(torch.load(args.output / f"{label}.pt", weights_only=True, map_location=device))
            test_values = evaluator(test, test_data, label == "full")
            values = scores_for(lambda rows: test_values(policy, rows), len(test.close), device)
            trace = replay(values, test, *test_data)
            traces[label] = trace
            results[label]["test"] = metrics(trace)
            np.savez_compressed(args.output / f"{label}_test.npz", **trace)
        full, control = traces["full"], traces["ledger_only"]
        same_position = full["previous_action"] == control["previous_action"]
        disagree = (full["votes"] != full["votes"][:, :1]).any(axis=1)
        results["test_comparison"] = {
            "same_position_expert_disagreement_minutes": int((same_position & disagree).sum()),
            "different_actions_with_same_position_and_expert_disagreement": int(
                (same_position & disagree & (full["action"] != control["action"])).sum()),
        }
    results["execution"] = {"seed": args.seed, "device": str(device), "limit": args.limit,
                            "elapsed_seconds": round(perf_counter() - started, 1)}
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(results, flush=True)


if __name__ == "__main__":
    main()
