# Fixed-entry exit experiment

The three frozen forecasts from the horizon signal study provide the entry score. A score of at least 2 bp triggers a simulated purchase at the next close when the previous 120-minute window has ended. The level is the existing one-way fee, selected as an experimental entry convention. An episode reserves the entire 120-minute window for every policy. This produces common entry events so early exits cannot add extra trades.

The strategy may first exit one minute after entry and must exit within 120 minutes. At every exit decision, observations use completed bars through the previous close. The policy receives the three forecasts, elapsed time and unrealized return as of that previous close. The simulator charges 2 bp on entry and exit. Entry costs and the first price interval are identical for all exits in an episode. To train Double Q, an exit action pays the remaining fee and terminates, while waiting collects the next price change and continues until a forced sale. The finite-horizon objective sums rewards with a discount of one. Both rewards use the entry notional as their denominator.

The fixed rules exit after 5, 30 or 120 minutes, or when the mean predicted return falls to minus 2 bp, with a 120-minute cap. Validation selects the fixed rule and the checkpoint of each Double-Q arm by total net PnL. The paired and sampled variants use the same network, seed, updates and entry episodes. The paired variant uses both exit and wait targets at each eligible minute, and the sampled variant uses one uniformly sampled action. Evaluation compares realized fees and PnL over the common entry schedule. Results refer to the previously examined historical test.

```bash
python -m eth_router.exit_study --source data/raw --output runs/fixed-entry-exit-v1 --device cuda:0
```
