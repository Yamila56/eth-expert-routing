"""Load an external checkout and batch the frozen expert evaluations."""
import importlib.util

import numpy as np
import torch

from .policy import EXPERTS


class ExpertSource:
    def __init__(self, checkout, device):
        self.device = torch.device(device)
        checkout = checkout.resolve()
        spec = importlib.util.spec_from_file_location("external_expert_network", checkout / "model/net.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        feature_dir = checkout / "data/feature_list"
        self.state_columns = np.load(feature_dir / "single_features.npy", allow_pickle=True).tolist()
        self.trend_columns = np.load(feature_dir / "trend_features.npy", allow_pickle=True).tolist()
        weights = checkout / "result/low_level/ETHUSDT/best_model"
        self.models = []
        for label in EXPERTS:
            group, index = label.split("_")
            model = module.subagent(len(self.state_columns), len(self.trend_columns), 2, 64)
            model.load_state_dict(torch.load(weights / group / index / "best_model.pkl",
                                             map_location="cpu", weights_only=True))
            self.models.append(model.to(self.device).eval().requires_grad_(False))

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
                    # The source centers advantages across its batch. Restore the
                    # per-observation centering used when acting one minute at a time.
                    adjustment = []
                    hook = expert.advantage.register_forward_hook(
                        lambda _model, _args, advantage: adjustment.append(
                            advantage.mean() - advantage.mean(dim=1, keepdim=True))
                    )
                    try:
                        q = expert(state, trend, previous) + adjustment[0]
                    finally:
                        hook.remove()
                    scores[window, prior, i] = q.cpu().numpy()
        return scores
