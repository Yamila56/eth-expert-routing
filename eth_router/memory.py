"""Random transition replay and a separate store of similar experiences."""
import numpy as np


TRANSITION_DTYPE = [
    ("minute", "i8"), ("position", "i8"), ("action", "i8"), ("reward", "f4"),
    ("terminal", "?"), ("memory_value", "f4"),
]


class TransitionReplay:
    def __init__(self, capacity):
        self.entries = np.empty(capacity, dtype=TRANSITION_DTYPE)
        self.total = 0

    def append(self, transition, memory_value):
        state = transition.observation
        self.entries[self.total % len(self.entries)] = (
            state.minute, state.position, transition.action, transition.reward,
            transition.terminal, memory_value,
        )
        self.total += 1

    def sample(self, batch_size, rng):
        indices = rng.integers(min(len(self.entries), self.total), size=batch_size)
        return self.entries[indices]


class EpisodicValueMemory:
    """Retrieve the value of a matching action in nearby encoded states."""

    def __init__(self, capacity=4320, neighbors=5):
        self.capacity = capacity
        self.neighbors = neighbors
        self.keys = None
        self.minutes = np.empty(capacity, dtype=np.int64)
        self.positions = np.empty(capacity, dtype=np.int64)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.values = np.empty(capacity, dtype=np.float64)
        self.total = 0
        self.hits = 0
        self.refreshes = 0

    def estimate(self, key, action, fallback):
        if self.total < self.capacity:
            return fallback
        distances = np.square(self.keys - key).sum(axis=1)
        nearest = np.argsort(distances)[:self.neighbors]
        matching = nearest[self.actions[nearest] == action]
        if not matching.size:
            return fallback
        inverse = 1 / (distances[matching] + 0.001)
        self.hits += 1
        return float(np.average(self.values[matching], weights=inverse))

    def append(self, observation, action, key, estimate):
        if self.keys is None:
            self.keys = np.empty((self.capacity, key.size), dtype=np.float64)
        index = self.total % self.capacity
        self.keys[index] = key
        self.minutes[index] = observation.minute
        self.positions[index] = observation.position
        self.actions[index] = action
        self.values[index] = estimate
        self.total += 1

    def refresh_keys(self, encode):
        size = min(self.capacity, self.total)
        for start in range(0, size, 512):
            window = slice(start, min(start + 512, size))
            self.keys[window] = encode(self.minutes[window], self.positions[window])
        self.refreshes += 1
