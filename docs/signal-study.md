# Horizon signal and position-policy study

This experiment evaluates horizon-specific forecasts and compares rules and value-based policies on the same information and execution model.

## Changes and rationale

| Proposal | Decision |
| --- | --- |
| Explain expert overrides | Compare baseline and full actions from the same incoming position. Use the starting position and both actions to identify an extra entry, withheld entry, early exit or delayed exit. Compare holding both target positions for 5, 30 and 120 minutes, including the entry switch fee. |
| Fixed residual scaling | Compare frozen policies at weights 0, 0.25, 0.5, 0.75 and 1 with a complete position replay for each weight. |
| Confidence-gated residual | Revisit after a signal shows economic value on a later period. A baseline Q gap is an action-value difference, and interpreting it as confidence requires calibration. |
| Horizon-specific inputs and targets | Fit three regularized predictors of 5-, 30- and 120-minute returns from the simulated fill. Their input windows are 2/5, 5/30 and 30/120 minutes respectively. Scaling and parameters use the first 60% of the training split. |
| Multi-step TD specialists | Defer while testing the signal directly. Using a multi-step target for trading requires specifying the continuation trades and their fees for each starting action. |
| Shorting, variable size and stops | Hold the execution task fixed for the experiment. Each strategy selects flat or long 0.2 ETH with the same 2 bp switch fee and next-bar close. |

## Shared input and policy comparisons

The three predictors are fixed before the last 40% of the training split. Policy training receives their forecasts as inputs. Signal diagnostics use the 90th percentile of each prediction on that training portion and report later returns after the cost of buying and selling. Adjacent holding windows overlap.

The rule trades on the mean of the three forecasts with a symmetric entry and exit buffer. Validation selects the buffer from 0, 2, 5, 10 and 20 bp plus a flat fallback. The two Double-Q arms take the same forecast set and incoming position. Each uses 12 epochs, the same seed and minibatch order, and the same TD objective. One samples a position transition per minute and the other evaluates all four. Validation net PnL picks the RL checkpoints. The run saves learning curves, selected checkpoints and historical test traces. Always-flat and buy-and-hold report economic reference points on the same fill and marking convention.

The historical test interval has been examined in earlier work. Its results support diagnosis and comparison on that interval. A fresh period is needed for confirmation beyond it.

```bash
python -m eth_router.signal_study --source data/raw --output runs/signal-study-v1 --device cuda:0
```
