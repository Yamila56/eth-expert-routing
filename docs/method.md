# Method

## Observation and fill timing

Minute observations through bar `t` determine a target position at the end of that bar. The simulator fills the order at the close of bar `t+1`. The incoming position was established by the previous order. The new position earns the price movement from the close of `t+1` to the close of `t+2`, with the switch fee charged at fill. The next decision uses the completed data through `t+1`. The replay stores the observation bar label and fill bar label separately. The one-bar delay gives the experiment a consistent convention even when the raw timestamps do not specify the availability of each source field within the minute. The fill price is a simulation assumption.

The learner's reward is the net change in basis points relative to the full-position notional at the execution close. For binary incoming position $p$, action $a$, fill price $P$ and next mark $P'$, it is $10^4 a(P'-P)/P - 2|a-p|$. The replay uses 0.2 ETH and evaluates the corresponding cash PnL in USDT. The two quantities serve different purposes. Summed training rewards do not represent a compounded portfolio return.

## Expert evidence

The three experts use reward discounts with half-lives of 5, 30 and 120 minutes and share the same model structure, data and initialization. A half-life specifies how quickly future rewards lose weight. Each expert makes a new decision every minute.

For each recommendation and starting position, the simulator compares taking the recommended action with taking the other action. Both resulting positions are held through the same 5-, 30- or 120-minute window, with the opening switch fee and the closing mark included. Each expert receives the same set of evaluation windows. A result can enter an observation only after the closing mark is available. The fixed holding rule gives all recommendations a common experiment. It measures the outcome of the recommendation under that rule and does not reproduce the expert's subsequent policy.

An independent simulation follows each expert's actual sequence of recommendations and supplies rolling net changes and turnover. The policy reads both types of history over the preceding 360 decision rows. Only settled intervals enter the ledger, and outcome records include their settled sample count.

## Policy and comparisons

Experts train on the first 60% of the training split and are frozen. The remaining 40% trains a market-only position network and the two expert corrections. The market network is selected on validation net PnL and frozen before the corrections are trained. Each correction starts at zero, and each uses the same frozen baseline and model structure. The full policy uses both evidence records. The ledger-only policy uses current recommendations, action margins and independent accounts, with the multiscale outcome fields masked.

All three policies use paired Double Q targets and Huber losses on action values and their difference. Validation net PnL selects each correction checkpoint. The within-position replay comparison measures action changes relative to the market baseline for the same incoming position. Separate trajectory replays capture what those changes do to the account over time. Validation serves checkpoint selection, so validation differences are descriptive.

The expert representations are pooled with a shared encoder. The ordered input is a standardized log half-life. It describes the expert's decision scale without giving each expert its own input column. The policy uses past recommendation outcomes as observations.
