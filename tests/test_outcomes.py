import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from eth_router.outcomes import decision_evidence, load_market, net_reward_bps


class DecisionEvidenceTest(unittest.TestCase):
    def test_starting_position_changes_fee_and_labels_appear_at_maturity(self):
        close = np.full(6, 100.0)
        opinions = np.ones((6, 2, 1), dtype=int)
        evidence = decision_evidence(close, opinions, horizons=(2, 3), history=2)
        np.testing.assert_array_equal(evidence[:3, ..., 2:5], 0)
        np.testing.assert_array_equal(evidence[:4, ..., 5:], 0)
        np.testing.assert_allclose(evidence[3, :, 0, 2:5], [[-2, 0, 1], [2, 1, 1]])
        np.testing.assert_allclose(evidence[4, :, 0, 5:], [[-2, 0, 1], [2, 1, 1]])
        np.testing.assert_array_equal(evidence[:, 0, 0, 4], [0, 0, 0, 1, 2, 2])
        np.testing.assert_array_equal(evidence[:2, :, 0, 0], 0)
        np.testing.assert_allclose(evidence[2, :, 0, 0], [-2, -2])

    def test_execution_bar_price_cannot_change_current_evidence(self):
        opinions = np.ones((8, 2, 1), dtype=int)
        prices = np.arange(100.0, 108.0)
        updated = prices.copy()
        updated[5:] += 50
        original = decision_evidence(prices, opinions, horizons=(2,))
        changed = decision_evidence(updated, opinions, horizons=(2,))
        np.testing.assert_array_equal(original[:6], changed[:6])
        self.assertFalse(np.array_equal(original[6:], changed[6:]))

    def test_execution_bar_is_excluded_from_market_observations(self):
        frame = pd.DataFrame({"timestamp": pd.date_range("2023-01-01", periods=125, freq="min"),
                              "close": 100.0, "volume": 1.0, "ask1_price": 101.0,
                              "bid1_price": 99.0, "volume_imbalance": 0.0})
        with TemporaryDirectory() as directory:
            source = Path(directory)
            path = source / "df_test.feather"
            frame.to_feather(path)
            time, _, features, _ = load_market(source, "test")
            frame.loc[122, ["close", "volume", "ask1_price", "volume_imbalance"]] = [120, 10, 121, 0.5]
            frame.to_feather(path)
            _, close, changed, _ = load_market(source, "test")
        np.testing.assert_equal(time[1], frame.timestamp.iloc[122].to_datetime64())
        self.assertEqual(close[1], 120)
        np.testing.assert_array_equal(features[:2], changed[:2])
        self.assertFalse(np.array_equal(features[2], changed[2]))

    def test_reward_is_invariant_to_price_level(self):
        np.testing.assert_allclose(
            net_reward_bps(np.array([100, 400]), np.array([101, 404]), 0, 1), [98, 98]
        )
