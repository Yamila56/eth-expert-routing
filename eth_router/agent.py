"""The high-level agent owns the online Q network, target network, and learning state."""
from copy import deepcopy

import numpy as np
import torch
from torch.nn import functional as F

from .memory import EpisodicValueMemory, TransitionReplay


class RoutingAgent:
    def __init__(self, router, inputs, expert_scores, demonstrations, capacity, seed, device):
        self.online = router
        self.target = deepcopy(router).requires_grad_(False)
        self.optimizer = torch.optim.Adam(router.parameters(), lr=1e-4)
        self.inputs = inputs
        self.expert_scores = expert_scores
        self.demonstrations = demonstrations
        self.device = torch.device(device)
        self.random = np.random.default_rng(seed)
        self.replay = TransitionReplay(capacity)
        self.memory = EpisodicValueMemory()
        self.discount = 0.99

    def _observe(self, minute, previous):
        context = tuple(self._tensor(array[minute]) for array in self.inputs)
        return (*context, self._tensor(previous))

    def _tensor(self, data):
        return torch.as_tensor(data, device=self.device)

    def _values(self, network, minute, previous):
        inputs = self._observe(minute, previous)
        return network(*inputs, self._tensor(self.expert_scores[minute, previous]))[0]

    @torch.no_grad()
    def encode(self, minute, previous):
        state, trend, _, position = self._observe(minute, previous)
        return self.online.encode(state, trend, position).cpu().numpy()

    @torch.no_grad()
    def act(self, observation, epsilon):
        if self.random.random() < epsilon:
            return int(self.random.integers(2))
        return self._values(self.online, [observation.minute], [observation.position]).argmax().item()

    @torch.no_grad()
    def observe(self, transition):
        state = transition.observation
        action = transition.action
        key = self.encode([state.minute], [state.position])[0]
        bootstrap = 0.0
        if not transition.terminal:
            bootstrap = self._values(self.online, [state.minute + 1], [action]).max().item()
        experience_value = transition.reward + self.discount * bootstrap
        memory_value = self.memory.estimate(key, action, experience_value)
        self.replay.append(transition, memory_value)
        self.memory.append(state, action, key, experience_value)

    def learn(self, batch_size):
        batch = self.replay.sample(batch_size, self.random)
        minute, position, action = (batch[name].copy() for name in ("minute", "position", "action"))
        chosen = self._tensor(action)[:, None]
        current = self._values(self.online, minute, position)
        selected = current.gather(1, chosen).squeeze(1)
        with torch.no_grad():
            best_next = self._values(self.online, minute + 1, action).argmax(dim=1, keepdim=True)
            next_value = self._values(self.target, minute + 1, action).gather(1, best_next).squeeze(1)
            td_target = self._tensor(batch["reward"].copy()) + self.discount * next_value * self._tensor(~batch["terminal"])
        td_loss = F.mse_loss(selected, td_target)
        memory_loss = F.mse_loss(selected, self._tensor(batch["memory_value"].copy()))
        teacher = self._tensor(self.demonstrations[minute, position])
        demo_loss = F.kl_div(current.log_softmax(dim=1), teacher.softmax(dim=1), reduction="batchmean")
        objective = td_loss + 0.5 * memory_loss + 5.0 * demo_loss
        self.optimizer.zero_grad()
        objective.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimizer.step()
        with torch.no_grad():
            for lagged, online in zip(self.target.parameters(), self.online.parameters(), strict=True):
                lagged.lerp_(online, 0.005)
        return np.array([td_loss.item(), memory_loss.item(), demo_loss.item()])
