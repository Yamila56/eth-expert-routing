"""Read trusted expert weights and feature names without executing source code."""
from pathlib import Path

import numpy as np
import torch

from .experts import ExpertQNetwork
from .policy import EXPERTS

# Prefixes in the original serialized tensors are used only at this boundary.
WEIGHT_KEYS = {
    "fc1": "market",
    "fc2": "trend",
    "embedding": "position",
    "adaLN_modulation.1": "condition",
    "value": "level",
    "advantage": "preference",
}


def load_expert_weights(model, path):
    stored = torch.load(path, map_location="cpu", weights_only=True)
    mapped = {}
    for key, value in stored.items():
        if key == "max_punish":
            continue  # The source checkpoint includes a scalar unused by inference.
        prefix, suffix = next(
            ((source, key[len(source):]) for source in WEIGHT_KEYS if key.startswith(source + ".")),
            (None, None),
        )
        if prefix is None:
            raise ValueError(f"Unexpected expert parameter {key!r} in {path}")
        mapped[WEIGHT_KEYS[prefix] + suffix] = value
    model.load_state_dict(mapped)


class ExpertSource:
    def __init__(self, feature_dir, checkpoint_paths, device):
        self.device = torch.device(device)
        self.state_columns = np.load(feature_dir / "single_features.npy", allow_pickle=True).tolist()
        self.trend_columns = np.load(feature_dir / "trend_features.npy", allow_pickle=True).tolist()
        self.models = []
        for path in checkpoint_paths:
            model = ExpertQNetwork(len(self.state_columns), len(self.trend_columns))
            load_expert_weights(model, path)
            self.models.append(model.to(self.device).eval().requires_grad_(False))

    @staticmethod
    def add_arguments(parser):
        path = parser.add_mutually_exclusive_group(required=True)
        path.add_argument("--expert-assets", type=Path)
        path.add_argument("--upstream", type=Path, help="Author checkout for weights and feature names only")

    @classmethod
    def from_args(cls, args, device):
        if args.expert_assets:
            return cls(args.expert_assets, [args.expert_assets / f"{name}.pkl" for name in EXPERTS], device)
        weights = args.upstream / "result/low_level/ETHUSDT/best_model"
        paths = [weights.joinpath(*name.split("_"), "best_model.pkl") for name in EXPERTS]
        return cls(args.upstream / "data/feature_list", paths, device)

    def market_inputs(self, frame):
        return tuple(frame[cols].to_numpy(dtype=np.float32, copy=True) for cols in (self.state_columns, self.trend_columns))

    @torch.inference_mode()
    def scores(self, frame, batch_size=4096):
        states, trends = self.market_inputs(frame)
        scores = np.empty((len(frame), 2, len(EXPERTS), 2), dtype=np.float32)
        for start in range(0, len(frame), batch_size):
            window = slice(start, start + batch_size)
            state = torch.as_tensor(states[window], device=self.device)
            trend = torch.as_tensor(trends[window], device=self.device)
            for prior in (0, 1):
                previous = torch.full((len(state),), prior, device=self.device, dtype=torch.long)
                for i, expert in enumerate(self.models):
                    scores[window, prior, i] = expert(state, trend, previous).cpu().numpy()
        return scores
