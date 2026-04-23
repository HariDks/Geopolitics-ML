# Backtest Results — Honest Numbers

## Two sets of numbers exist for this project

There are reported (in-distribution) numbers and out-of-sample (blind evaluation) numbers. Both are real — they measure different things. The out-of-sample numbers are what matters for real-world performance.

## Out-of-Sample Performance (the honest numbers)

Evaluated on data the model has never seen during training, using temporal holdout (train on pre-2023, test on 2023+) and blind evaluation datasets.

| Component | Out-of-Sample | How Measured |
|-----------|:------------:|:------------:|
| Event classifier | **64%** on unseen event types | Temporal holdout, novel event descriptions |
| Channel prediction (with text) | **62.3% top-2** | 70 blind eval pairs, frozen before evaluation |
| Channel prediction (without text) | **46.4% top-2** | Same 70 blind eval pairs |
| Impact interval coverage | **43%** on holdout | 163 manual labels, temporal split |
| Direction accuracy (+/-) | **86-90%** | Consistent across all evaluations |
| Negative detection (unaffected companies) | **100%** (10/10) | 10 companies with no event connection |

## In-Distribution Performance (reported during development)

These numbers were computed on validation sets that share distribution with training data. They are higher because the model is tested on familiar patterns.

| Component | In-Distribution | Why It's Higher |
|-----------|:--------------:|:----------------|
| Event classifier | 95.3% | Tested on same text formats as training (ACLED, GTA) |
| Channel prediction | 82.5% macro F1 | Validation includes auto-generated labels (feedback loop) |
| Channel prediction (with lexicon) | 95.2% | Cross-company validated but same events as training |
| Impact interval coverage | 80.7% | Event studies cluster near zero, easy to cover |

## The gap explained

The gap between in-distribution and out-of-sample is not fraud — it's a well-known phenomenon in ML called **distribution shift**. The model learns patterns from its training data. When test data comes from the same distribution (same sources, same time period, same event types), accuracy is high. When test data comes from genuinely novel situations, accuracy drops.

The out-of-sample numbers are what a real user would experience.

## Key backtest findings

- **Direction accuracy is the model's genuine strength.** 86-90% across all evaluations, validated by stock prices.
- **Channel prediction works well with descriptive text** (62.3% top-2) but struggles without it (46.4%).
- **Impact magnitude is the weakest component.** The model estimates generic patterns, not firm-specific impacts.
- **The model correctly identifies unaffected companies** (100% on negative cases).

## Detailed backtest on 10 historical events

See `backtest/run_backtest.py` for the full 10-event, 14-company-pair backtest. Best predictions:

| Company | Event | Predicted | Actual | Gap |
|---------|-------|:---------:|:------:|:---:|
| NVIDIA | Chip controls | -6.3% | -5.0% | 1.3pp |
| KLAC | Chip controls | -14.6% | -15.0% | 0.4pp |
| First Quantum | Panama mine | -12.2% | -13.1% | 0.9pp |
| Maersk | Red Sea | +13.8% | +15.0% | 1.2pp |

Worst predictions:

| Company | Event | Predicted | Actual | Gap |
|---------|-------|:---------:|:------:|:---:|
| Treasury Wine | China tariff | ~0% | -96% | 96pp |
| Zain Group | Sudan war | -0.5% | -71% | 70pp |

The failures are concentrated in **niche companies with extreme geographic concentration** — the model lacks firm-specific exposure data.
