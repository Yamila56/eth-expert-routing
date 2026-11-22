# Forecast-driven position rule

This historical case study uses 5-, 30- and 120-minute forecasts to choose between a flat account and a position of 0.2 ETH. Validation selects the trading buffer from a fixed grid. The selected buffer was 20 bp, plus the 2 bp one-way fee. Decisions use completed bars and fill at the following close. Open positions are marked at the final close.

| Strategy | Net PnL (USDT) | Maximum drawdown (USDT) | Position switches |
| --- | ---: | ---: | ---: |
| Always flat | 0.00 | 0.00 | 0 |
| Buy and hold | -11.97 | 100.23 | 1 |
| Forecast-driven rule | +45.49 | 45.19 | 1 |

The June through October 2023 replay contains 220,198 decision minutes. The rule bought once on August 17, at 1,586.92 USDT, and held through the final mark of 1,814.67 USDT. The result comes from that single entry and subsequent exposure. These figures describe a previously examined interval and apply to the deterministic rule. The [shared-signal policy protocol](signal-study.md) describes the separate Double Q comparison.

## Reproduce the replay

Place the three ETH/USDT Feather splits described in the [Quick Start](../README.md#quick-start) in `data/raw/`.

```bash
python -m eth_router.signal_study --source data/raw --output runs/rule-replay --rule-only
```

The command saves validation and replay metrics in `metrics.json` and minute-level records for all three strategies.
