import unittest

import numpy as np

from eth_router.market_env import Observation, TradingEpisode, hindsight_values, settle
from eth_router.memory import EpisodicValueMemory, TransitionReplay


class AgentTrainingTest(unittest.TestCase):
    def test_settlement_and_demonstration(self):
        prices = np.array([100.0, 104.0, 97.0])
        episode = TradingEpisode(prices)
        transition = episode.step(1)
        self.assertAlmostEqual(transition.reward, 0.796)
        self.assertFalse(transition.terminal)
        self.assertTrue(episode.step(0).terminal)

        teacher = hindsight_values(prices, 0.99)
        for previous in (0, 1):
            for action in (0, 1):
                first = settle(prices[0], prices[1], previous, action, 0.001)[0]
                last = max(settle(prices[1], prices[2], action, end, 0.001)[0] for end in (0, 1))
                self.assertAlmostEqual(teacher[0, previous, action], first + 0.99 * last, places=6)

    def test_memory_retrieves_nearest_matching_action(self):
        memory = EpisodicValueMemory(capacity=3, neighbors=2)
        key = np.array([0.0])
        self.assertEqual(memory.estimate(key, 1, -1), -1)
        for minute, (location, action, value) in enumerate([(0.1, 0, 8), (0.2, 1, 3), (50, 1, 80)]):
            memory.append(Observation(minute, 0), action, np.array([location]), value)
        self.assertAlmostEqual(memory.estimate(key, 1, -1), 3)
        self.assertEqual(memory.hits, 1)
        memory.refresh_keys(lambda minutes, positions: minutes[:, None].astype(float))
        self.assertEqual(memory.refreshes, 1)
        np.testing.assert_array_equal(memory.keys[:, 0], [0, 1, 2])

    def test_transition_replay_discards_oldest_entry(self):
        episode = TradingEpisode([100, 101, 102, 103])
        replay = TransitionReplay(capacity=2)
        for action in (0, 1, 0):
            replay.append(episode.step(action), 0.0)
        sample = replay.sample(100, np.random.default_rng(27))
        self.assertEqual(set(sample["minute"]), {1, 2})
