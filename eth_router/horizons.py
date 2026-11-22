"""Train and evaluate three horizon experts and an independent position policy."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from .market_env import settle
from .outcomes import (
    ACCOUNT_FIELDS, HALF_LIVES, HISTORY_MINUTES, OUTCOME_FIELDS, OUTCOME_WINDOWS,
    decision_evidence, load_market, net_reward_bps, position_path,
)
from .paired import fit_action_values


class HorizonExpert(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.values = nn.Sequential(nn.Linear(features + 1, 48), nn.SiLU(),
                                    nn.Linear(48, 48), nn.SiLU(), nn.Linear(48, 2))

    def forward(self, market, position):
        return self.values(torch.cat((market, position), dim=-1))


class PositionPolicy(nn.Module):
    def __init__(self, market_prior):
        super().__init__()
        self.market_prior = market_prior.requires_grad_(False)
        width = 3 + ACCOUNT_FIELDS + OUTCOME_FIELDS * len(OUTCOME_WINDOWS)
        self.expert_encoder = nn.Sequential(nn.Linear(width, 32), nn.SiLU(), nn.Linear(32, 32))
        self.residual = nn.Sequential(nn.Linear(33, 48), nn.SiLU(), nn.Linear(48, 2))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, market, position, evidence):
        encoded = self.expert_encoder(evidence).mean(dim=-2)
        change = self.residual(torch.cat((position, encoded), dim=-1))
        return self.market_prior(market, position) + change


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
    # One normalized log half-life per expert. The shared encoder is independent of set size.
    log_horizons = np.log2(HALF_LIVES)
    horizon = np.broadcast_to((log_horizons - log_horizons.mean()) / log_horizons.std(), opinions.shape)
    records = evidence.copy()
    records[..., 0] /= 100
    records[..., ACCOUNT_FIELDS::OUTCOME_FIELDS] /= 100
    records[..., ACCOUNT_FIELDS + 2::OUTCOME_FIELDS] /= HISTORY_MINUTES
    if not include_outcomes:
        records[..., ACCOUNT_FIELDS:] = 0
    result = np.concatenate((horizon[..., None], opinions[..., None], margins[..., None], records),
                            axis=-1, dtype=np.float32)
    return torch.as_tensor(result, device=device)

def replay(values, market, opinions, margins, evidence, *, prior_values):
    previous, action = position_path(values[:-1].argmax(axis=-1)[..., None])
    previous, action = previous[:, 0], action[:, 0]
    reward, fee, _ = settle(market.close[:-1], market.close[1:], previous, action)
    row = np.arange(len(action))
    return {"timestamp": market.time[:-1] - np.timedelta64(1, "m"),
            "execution_timestamp": market.time[:-1], "outcome_windows": np.asarray(OUTCOME_WINDOWS),
            "action": action, "previous_action": previous, "market_q": prior_values[row, previous],
            "action_q": values[row, previous], "votes": opinions[row, previous],
            "margins": margins[row, previous], "evidence": evidence[row, previous],
            "reward": reward, "fee": fee, "half_lives": np.asarray(HALF_LIVES)}


def metrics(trace):
    equity = np.r_[0, trace["reward"].cumsum()]
    votes = trace["votes"]
    return {"minutes": len(votes), "net_pnl_usdt": float(equity[-1]),
            "max_drawdown_usdt": float(np.max(np.maximum.accumulate(equity) - equity)),
            "fees_usdt": float(trace["fee"].sum()),
            "expert_changes_vs_prior_same_position": int(np.count_nonzero(
                trace["market_q"].argmax(axis=-1) != trace["action"])),
            "pairwise_expert_disagreement": {
                f"{HALF_LIVES[i]}_vs_{HALF_LIVES[j]}": float((votes[:, i] != votes[:, j]).mean())
                for i in range(len(HALF_LIVES)) for j in range(i + 1, len(HALF_LIVES))
            },
            "switches": int(np.count_nonzero(trace["action"] != trace["previous_action"])),
            "disagreement_minutes": int(np.count_nonzero((votes != votes[:, :1]).any(axis=1)))}


def validation_td_error(values, close, discount=0.99):
    position = np.arange(2)
    rewards = net_reward_bps(close[:-1, None, None], close[1:, None, None],
                             position[None, :, None], position[None, None, :])
    continuation = values[1:].max(axis=-1).copy()
    continuation[-1] = 0
    absolute_error = np.abs(values[:-1] - (rewards + discount * continuation[:, None, :]))
    return float(np.where(absolute_error < 1, 0.5 * absolute_error**2, absolute_error - 0.5).mean())


def expert_disagreement(opinions):
    labels = ("flat", "long")
    return {
        label: {
            f"{HALF_LIVES[i]}_vs_{HALF_LIVES[j]}": float((opinions[:-1, p, i] != opinions[:-1, p, j]).mean())
            for i in range(len(HALF_LIVES)) for j in range(i + 1, len(HALF_LIVES))
        }
        for p, label in enumerate(labels)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, help="Use the first N raw rows per split for a smoke run")
    parser.add_argument("--expert-epochs", type=int, default=12)
    parser.add_argument("--policy-epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--test", action="store_true", help="Evaluate the frozen market prior and both expert policies on the test split")
    parser.add_argument("--training-ablation", action="store_true",
                        help="Compare sampled and paired TD updates on the full expert policy")
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
                                             discount=2 ** (-1 / half_life), reward_bps=net_reward_bps):
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
    curves = {}

    def fit_policy(label, policy, training_values, validation_values, prior_values=None, **training_options):
        best = -np.inf
        checkpoint = args.output / f"{label}.pt"
        curve = []
        for epoch, loss in fit_action_values(policy, training_values, policy_train.close,
                                             epochs=args.policy_epochs, seed=args.seed, device=device,
                                             reward_bps=net_reward_bps, **training_options):
            values = scores_for(lambda rows: validation_values(policy, rows), len(validation.close), device)
            trace = replay(values, validation, *validation_data,
                           prior_values=values if prior_values is None else prior_values)
            summary = metrics(trace)
            if args.training_ablation:
                curve.append({"epoch": epoch, **loss, "net_pnl_usdt": summary["net_pnl_usdt"],
                              "validation_bellman_huber": validation_td_error(values, validation.close)})
            print({"policy": label, "epoch": epoch, **loss, **summary,
                   "elapsed_seconds": round(perf_counter() - started, 1)}, flush=True)
            if summary["net_pnl_usdt"] > best:
                best = summary["net_pnl_usdt"]
                results[label] = {"validation": summary, "selected_epoch": epoch}
                torch.save(policy.state_dict(), checkpoint)
                np.savez_compressed(args.output / f"{label}_val.npz", **trace)
        if curve:
            curves[label] = curve
        policy.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))
        return policy.eval()

    torch.manual_seed(args.seed)
    prior = fit_policy("market_only", HorizonExpert(len(columns)).to(device),
                       policy_train.values, validation.values).requires_grad_(False)
    validation_prior = scores_for(lambda rows: validation.values(prior, rows), len(validation.close), device)
    policies = {}
    for label, include_outcomes in (("full", True), ("ledger_only", False)):
        torch.manual_seed(args.seed)
        policies[label] = fit_policy(label, PositionPolicy(prior).to(device),
                                     evaluator(policy_train, training_data, include_outcomes),
                                     evaluator(validation, validation_data, include_outcomes),
                                     validation_prior)

    if args.training_ablation:
        full_training = evaluator(policy_train, training_data, True)
        full_validation = evaluator(validation, validation_data, True)
        for label, sampled in (("paired_td", False), ("sampled_td", True)):
            torch.manual_seed(args.seed)
            policies[label] = fit_policy(label, PositionPolicy(prior).to(device),
                                         full_training, full_validation, validation_prior,
                                         sampled=sampled, margin_weight=0.0)
        results["learning_curves"] = curves
        results["expert_disagreement"] = {"validation": expert_disagreement(validation_data[0])}

    if args.test:
        test = load_split("test")
        test_data = prepare(test)
        if args.training_ablation:
            results["expert_disagreement"]["test"] = expert_disagreement(test_data[0])
        test_prior = scores_for(lambda rows: test.values(prior, rows), len(test.close), device)
        values_by_label = {"market_only": test_prior}
        for label, policy in policies.items():
            test_values = evaluator(test, test_data, label != "ledger_only")
            values_by_label[label] = scores_for(lambda rows: test_values(policy, rows), len(test.close), device)

        traces = {}
        for label, values in values_by_label.items():
            trace = replay(values, test, *test_data, prior_values=test_prior)
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
                            "execution_lag_bars": 1, "learning_reward": "basis_points",
                            "outcome_windows": OUTCOME_WINDOWS,
                            "elapsed_seconds": round(perf_counter() - started, 1)}
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(results, flush=True)


if __name__ == "__main__":
    main()
