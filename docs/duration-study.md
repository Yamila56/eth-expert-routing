# Contextual holding-period selection

This final diagnostic uses the existing fixed entry schedule. At each entry the policy chooses a 5-, 30-, 60- or 120-minute holding period. The entry decision sees completed bars through the bar before the fill. The reserved window is identical across all periods, so a choice leaves the subsequent entry schedule intact.

A ridge model predicts all four net outcomes in basis points. Its inputs are the three frozen forecasts and volatility, spread and imbalance at 5, 30 and 120 minutes. Its regularization is fixed at 0.1 per training observation. It trains on the second portion of the original training split. The strongest fixed duration is selected by net USDT PnL on validation, with ties resolved in ascending duration order. The previously examined test period serves as a historical comparison. Drawdown includes minute-by-minute mark-to-market movements and both trade fees.

This is a full-feedback contextual decision experiment. The existing fixed-entry Double Q study separately examines sequential exit decisions.

```bash
python -m eth_router.duration_study --source data/raw --output runs/holding-duration-v1
```
