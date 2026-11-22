import unittest

import numpy as np

from eth_router.outcomes import decision_evidence


class DecisionEvidenceTest(unittest.TestCase):
    def test_starting_position_changes_fee_and_labels_appear_at_maturity(self):
        close = np.full(5, 100.0)
        opinions = np.ones((5, 2, 1), dtype=int)
        evidence = decision_evidence(close, opinions, horizon=2, history=2)
        np.testing.assert_array_equal(evidence[:2, ..., 2:], 0)
        np.testing.assert_allclose(evidence[2, :, 0, 2:], [[-2, 0, 1], [2, 1, 1]])
        np.testing.assert_array_equal(evidence[:, 0, 0, 4], [0, 0, 1, 2, 2])
        np.testing.assert_allclose(evidence[1, :, 0, 0], [-2, -2])

    def test_future_price_does_not_change_current_evidence(self):
        opinions = np.ones((8, 2, 1), dtype=int)
        prices = np.arange(100.0, 108.0)
        updated = prices.copy()
        updated[5:] += 50
        original = decision_evidence(prices, opinions, horizon=2)
        changed = decision_evidence(updated, opinions, horizon=2)
        np.testing.assert_array_equal(original[:5], changed[:5])
        self.assertFalse(np.array_equal(original[5:], changed[5:]))
