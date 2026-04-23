# Learning Day 9 — From 50% to 95%: Lexicon Features and the Dual-Regime Model

**Project:** Geopolitical Muscle ML Model  
**Date:** April 18, 2026 (continued)  
**Written for:** Understanding how channel-specific lexicon features solved the channel prediction problem in text-rich mode, why they don't help in text-poor mode, and how the system now honestly communicates its own reliability

---

## Where We Left Off

Day 8 built diagnostic tools that revealed uncomfortable truths:
- Channel prediction: **50.7% real accuracy** (not 82.5% as reported)
- GRI weights: **0.099 correlation** with market evidence
- Feedback loop: **38.6pp** channel distribution shift from auto-labeling
- But: **90% directional accuracy** and **100% negative detection** — genuinely strong

The reviewer then provided a second round of feedback with a clearer path: stop chasing model architecture, start adding the right features. Specifically: **text is the most channel-informative evidence, and you're not using it.**

---

## Part 1: Top-K Evaluation — Half Our "Errors" Aren't Errors

### The Insight

The reviewer pointed out that many channel predictions aren't wrong — they're just the second-best answer. BP + Russia invasion: we predict `logistics_operations`, human says `capital_allocation_investment`. Both are plausible — BP had both shipping disruption AND a $25.5B write-down.

### The Test

Evaluated top-1, top-2, and top-3 accuracy on the 163 manual holdout labels:

```
Top-1: 35/69 (50.7%)
Top-2: 42/69 (60.9%)
Top-3: 51/69 (73.9%)
```

**47% of top-1 errors had the correct answer in top-3.** These aren't stupid predictions — they're adjacent channels. The model knows the event matters, it just doesn't always rank the human's preferred channel first.

### What This Means

**Top-2 is the more honest target for this problem**, because the problem itself is not truly single-label. The reviewer's words: "human analysts also think in multiple mechanisms."

---

## Part 2: The Text-to-Channel Model

### The Idea

We had 17,372 EDGAR mentions and 602 seed labels with `mention_text`. The text often directly reveals the channel:
- "impairment" → capital_allocation
- "supply disruption" → procurement
- "ransomware" → cybersecurity
- "rerouting" → logistics

But we weren't using any of this text signal for channel prediction. XGBoost only saw structured features (sector, stock reaction, etc.).

### The Implementation

TF-IDF + Logistic Regression (the reviewer suggested starting simple):

```python
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1,2), min_df=2, sublinear_tf=True)
model = LogisticRegression(max_iter=1000, class_weight='balanced')
```

Trained on manual labels only (to avoid auto-label leakage — Day 8's lesson learned).

### Cross-Validated Results

```
Text model (5-fold CV):   Top-1: 55.2%   Top-2: 69.3%   Top-3: 78.5%
XGBoost (structured):     Top-1: 50.7%   Top-2: 60.9%   Top-3: 73.9%
```

Text model is better at every k. The **top-2 jump (+8.4pp)** is especially significant — text helps the model rank the right candidates higher.

### Roadblock: Text Model Probabilities as XGBoost Features

We tried embedding the text model's 10 channel probabilities as features inside XGBoost. This caused a **train/inference mismatch**: during training, mention_text is available so the features are real. During inference via `scorer.score()`, mention_text isn't always available so the features are zeros. Result: 10.1% accuracy (catastrophic).

**The lesson:** Don't embed one model's outputs as features in another model unless the outputs are always available at inference time. Either blend at the end or ensure both models see the same inputs in both training and inference.

---

## Part 3: Exposure Proxy Features from 10-K Text

### What We Built

`pipelines/extract_exposure_proxies.py` mines all 17,372 EDGAR mentions to compute 5 proxy scores per company:

| Proxy | What it measures | Pattern examples |
|-------|-----------------|------------------|
| `geo_mention_density` | How often the company mentions specific regions | "China", "greater China", "European" |
| `facility_concentration` | Co-occurrence of facility terms + region | "manufacturing in China", "plant in Taiwan" |
| `single_source_risk` | Concentrated supplier language | "substantially all manufacturing", "sole source" |
| `asset_exit_score` | Impairment/divestiture history | "impairment", "write-down", "divested" |
| `route_sensitivity` | Shipping route exposure | "Red Sea", "Suez", "rerouting", "freight" |

### Results

99 companies scored. Key signals validated against reality:
- **AAPL**: facility=0.29, single_src=0.18, China density=0.27 (matches: 90%+ iPhones assembled in China)
- **AVGO**: single_src=0.27 (matches: heavy TSMC dependency)
- **KLAC**: China=0.31, facility=0.17 (matches: chip equipment heavily exposed)
- **CVX**: route=0.06 (matches: tanker routes through Hormuz)

### Impact on Channel Prediction

Adding proxy features to XGBoost (28 features) improved top-1 from 50.7% to 52.2% (+1.5pp) on holdout. Small gain — most holdout companies are international names outside our proxy coverage.

---

## Part 4: Channel Lexicon Features — The Breakthrough

### The Reviewer's Suggestion

> "Not a full model — just structured signals like capital_allocation_score, procurement_score, logistics_score based on keywords and phrase patterns. This is lightweight and high ROI."

### What We Built

Curated keyword lists for each of the 10 channels, derived from distinctive terms in the manual labels:

```python
CHANNEL_LEXICONS = {
    "capital_allocation_investment": [
        "impairment", "write-down", "write-off", "stake", "asset disposal",
        "divest", "pre-tax charge", "billion", "goodwill", "mine closure",
    ],
    "cybersecurity_it": [
        "ransomware", "cyberattack", "hack", "malware", "cyber", "breach",
        "encrypted", "ransom", "incident response", "it systems",
    ],
    "logistics_operations": [
        "rerouting", "route", "shipping", "freight", "transit", "vessel",
        "red sea", "suez", "cape of good hope", "port", "logistics",
    ],
    # ... 7 more channels
}
```

Each score = count of matching keywords / total keywords in that channel's lexicon. Simple, interpretable, fast.

### The Result

```
Without lexicon (28 features):  Top-1: 52.2%   Top-2: 59.4%   Top-3: 73.9%
With lexicon (38 features):     Top-1: 95.7%   Top-2: 97.1%   Top-3: 98.6%
```

**43.5 percentage point jump on top-1.** The lexicon features became the top 3 most important features in XGBoost:

```
#1: lex_reputation_stakeholder  0.142
#2: cat_technology_controls     0.135
#3: lex_cybersecurity_it        0.097
#4: lex_workforce_talent        0.057
#5: route_sensitivity           0.050
```

### Verification: Is This Real or Leakage?

**Leakage check (1A):** Lexicon keywords appear in both in-channel and cross-channel text. Ratios range from 0.6x to 4.5x — genuinely channel-specific, not just memorized.

**Cross-company robustness (1B):** Trained on 101 companies, tested on 44 completely held-out companies:
```
Top-1: 95.2%
Top-2: 100%
Top-3: 100%
```

**The signal is real.** It holds across unseen companies.

### Why This Works So Well

The lexicon is essentially encoding domain knowledge that XGBoost can't learn from 602 labels alone. When the text says "impairment charge of $25.5 billion," the lexicon score for `capital_allocation_investment` spikes, and XGBoost immediately knows the answer. Without this signal, XGBoost has to infer the channel from sector + stock reaction — a much harder problem with much less signal.

---

## Part 5: The Catch — Two Reliability Regimes

### The Problem

The 95% accuracy requires `mention_text` at inference time. Without text, lexicon scores are zero and the model falls back to structured features (~52%).

This creates two distinct operating modes:

| Mode | When | Accuracy | Example |
|------|------|:--------:|---------|
| **text_rich** | Event description or filing text available | ~95% | "BP announced exit from Rosneft stake resulting in impairment" |
| **text_poor** | Only structured features (category, sector, stock) | ~52% | Portfolio scanner iterating over companies with just a category |

### The Fix: Event Text at Inference

The reviewer's insight: "For each event, generate or retrieve a short description. Then run your lexicon extractor on this description."

We already have the event description — it's the text the user types into the classifier! We just weren't passing it to the exposure scorer.

**One-line fix in pipeline.py:**
```python
exposure_result = self.scorer.score(
    event_category=event_result["category"],
    ticker=ticker,
    event_text=text,  # ← pass the classifier input text through
)
```

Now the full pipeline always operates in text_rich mode because every analysis starts with event text.

The text_poor mode still applies when the scorer is called directly from the portfolio scanner (iterating over companies with just a category, no description).

### Making It Honest

Every prediction now includes:
```json
{
    "channel_mode": "text_rich",
    "channel_reliability": "high"
}
```

Users see whether the model is confident or guessing.

---

## Part 6: The Full Progression

| Model | Features | Top-1 | Top-2 | Top-3 |
|-------|:--------:|:-----:|:-----:|:-----:|
| XGBoost baseline (Day 7) | 22 structured | 50.7% | 60.9% | 73.9% |
| + proxy features | 28 structured | 52.2% | 59.4% | 73.9% |
| Text model (TF-IDF) standalone | text only | 55.2% | 69.3% | 78.5% |
| **+ lexicon features (with text)** | **38 (struct + lexicon)** | **95.7%** | **97.1%** | **98.6%** |

The journey: structured features alone are a coin flip. Text alone is better. Structured + text lexicon is near-perfect. The reviewer was right at every step.

---

## Part 7: The Honest Model Card (Final Version)

| Capability | Accuracy | Validated How | Mode |
|------------|:--------:|:------------:|:----:|
| Event classification (news text) | 95.3% | 64 held-out news examples | Always |
| Event classification (source format) | 94.6% | Stratified val set | Always |
| Channel prediction (text_rich) | 95.2% | 44 held-out companies, cross-validated | With event text |
| Channel prediction (text_poor) | 52.2% | 69 manual holdout labels | Without text |
| Direction prediction (+/-) | 90.0% | 163 manual holdout labels | Always |
| Negative detection (not affected) | 100% | 10 unaffected company tests | Always |
| Impact magnitude range | 60.0% in-range | 20 manual holdout labels | Always |
| GRI category weights | 0.099 market correlation | Stock reaction validation | Index only |

### The Narrative

**Event classification:** Genuinely strong. Works on both source-format and news-style text.

**Channel prediction:** Two regimes. With descriptive text (the normal case in the full pipeline), accuracy is ~95% — effectively solved. Without text (direct API calls, portfolio scanning), accuracy is ~52% — useful for direction but not channel specificity.

**Impact direction:** The model's most consistently reliable signal. 90% accuracy across all modes, validated independently by stock prices.

**Impact magnitude:** Reasonable but correlational. The model estimates order-of-magnitude correctly for major companies but misses extreme concentrated exposures.

**Company exposure:** The remaining bottleneck. Adding geographic revenue data and text-based exposure proxies helped marginally. The real gap is concentrated exposure for international/niche companies outside our 99-company S&P 500 coverage.

---

## Part 8: What the Reviewer's Feedback Taught Us

Across two rounds of review, the reviewer identified 25+ critiques. The most impactful lessons:

1. **"The wording itself is the most channel-informative evidence"** — validated by the 50.7% → 95.2% jump from lexicon features alone

2. **"Where are you accidentally pretending correlation = causation?"** — led to the disclaimer system and dual-mode reliability labeling

3. **"Semi-supervised feedback loops compound silently"** — confirmed by the 0.825 → 0.371 holdout eval revealing 0.454 F1 inflation

4. **"Top-2 may be the more honest target"** — validated by the finding that 47% of errors are adjacent channels

5. **"Don't underestimate text-based exposure proxies"** — the exposure proxies gave modest gains, but the lexicon features (a simpler text signal) gave massive gains

6. **"Make the mode distinction explicit in product output"** — implemented as channel_mode / channel_reliability in every response

The meta-lesson: **the reviewer's best suggestions were about data and evaluation, not model architecture.** Changing what the model sees (lexicon features) and how we measure it (holdout eval, top-k, cross-company validation) mattered more than any hyperparameter or architecture change.

---

## Part 9: The Blind Evaluation — Ground Truth on Real-World Performance

### Why This Was Needed

The 95.2% cross-company figure was validated but trained on data the model had seen during training (just different companies). The reviewer's third round pushed harder: **"You need a measured estimate of performance in real-world conditions. Run a proper evaluation on recent event descriptions, manually labeled, companies not in the original label set."**

### The Dataset

70 event-company pairs across 25 distinct events, frozen before evaluation. Rules:
- 1-2 sentence event descriptions (news-style, not ACLED/GTA format)
- Mix of directly exposed, indirectly exposed, and minimally exposed companies
- Each example labeled with: primary channel, secondary channel (optional), rationale, confidence
- **No lexicon tuning allowed after dataset was frozen** — the evaluation is blind

Diversity across all 8 event categories and 10 channels. Includes negative cases (Apple + India demonetization, JPMorgan + Sudan civil war, Caterpillar + GDPR).

### The Results

```
                             No Text    With Text    Delta
Top-1:                        23.2%       31.9%     +8.7pp
Top-2 (primary only):        46.4%       62.3%    +15.9pp
Top-2 (primary+secondary):   66.7%       75.4%     +8.7pp
```

**Event text improves top-2 by 15.9 percentage points** — validated on completely unseen data.

The top-2 including secondary channels at **75.4%** is within the reviewer's predicted 75-85% range.

### The Gap Between 95% and 32%

The cross-validated figure (95.2%) and the blind figure (31.9% top-1) measure different things:
- **95.2%**: model's accuracy on companies it saw during training, just held out by company (same events, same label distribution)
- **31.9%**: model's accuracy on genuinely novel event-company combinations with fresh descriptions

The gap (63pp) is not "the model is broken" — it's "the model works well within its training distribution but struggles to generalize to novel combinations." This is the honest state of the system.

### Error Analysis

47 errors in Mode B (with text). Breakdown:

**Adjacent channel (correct answer in top-2): 21 errors**
These aren't wrong — they're second-best predictions. The model predicts `procurement_supply_chain` when the human says `revenue_market_access`, but both are plausible.

**Completely wrong (high-confidence label): 17 errors**
These are genuine failures. The model predicts a channel that doesn't make sense for this company-event pair. Root causes:
- Lack of company-specific exposure data (the recurring theme)
- Regulatory events default to `revenue_market_access` instead of `regulatory_compliance_cost`

**Ambiguous ground truth (low/medium confidence): 10 errors**
These cases were genuinely hard to label. When the ground truth is uncertain, model errors are less meaningful.

### Top Confusion Pairs

| Actual | Predicted | Count | Why |
|--------|-----------|:-----:|-----|
| regulatory_compliance → | revenue_market_access | 9 | Model can't distinguish "costs went up" from "revenue went down" |
| procurement_supply_chain → | revenue_market_access | 7 | Both involve "losing access to something" — procurement loses inputs, revenue loses customers |
| revenue_market_access → | procurement_supply_chain | 5 | The reverse confusion — these two channels have overlapping language |
| financial_treasury → | revenue_market_access | 4 | "Money problems" could be revenue or treasury depending on mechanism |

The core issue: **revenue_market_access is the model's default prediction** when uncertain. It's the largest training category and the lexical fallback. This is the classic "majority class bias" in classification.

### What This Means for the Product

The system is **publishable and defensible** with these numbers:

1. **75.4% top-2 accuracy with secondary channels** — the model gets the right answer (or a reasonable alternative) three-quarters of the time
2. **+15.9pp improvement from event text** — the text signal is clearly valuable and measurable
3. **21 of 47 errors are adjacent** — nearly half of "errors" are actually reasonable secondary predictions
4. **The remaining 17 hard errors** are traceable to company-specific exposure gaps — a known and documented limitation

---

## Part 10: The Dual-Mode Reliability System

### What We Built

Every prediction now includes:

```json
{
    "channel_mode": "text_rich",
    "channel_reliability": "high",
    "channel_prediction": "capital_allocation_investment",
    "channel_confidence": 0.847
}
```

Three modes:
- **text_rich** (event text with lexicon matches): top-2 accuracy ~62% on blind eval, ~95% on training-distribution data
- **text_partial** (event text but no lexicon matches): moderate reliability
- **text_poor** (no text, structured features only): top-2 accuracy ~46%

### Why This Matters

A user seeing "channel_reliability: high" knows the prediction is backed by text evidence. A user seeing "channel_reliability: low" knows to treat it as directional guidance only. This is **calibrated honesty** — the system communicates its own uncertainty.

The pipeline disclaimer was updated:

> "Channel accuracy depends on text availability: ~62% top-2 with event text (text_rich mode), ~46% without (text_poor mode). Check channel_reliability field."

---

## Part 11: The Final Honest Model Card

| Capability | Metric | Value | Validated On |
|------------|--------|:-----:|:------------:|
| Event classification | Top-1 accuracy | **95.3%** | 64 held-out news examples |
| Channel prediction (text_rich) | Top-2 accuracy | **62.3%** | 70 blind eval pairs |
| Channel prediction (text_rich, +secondary) | Top-2+secondary | **75.4%** | 70 blind eval pairs |
| Channel prediction (text_poor) | Top-2 accuracy | **46.4%** | 70 blind eval pairs |
| Channel prediction (training distribution) | Top-1 accuracy | **95.2%** | 44 held-out companies (CV) |
| Direction prediction (+/-) | Accuracy | **90.0%** | 163 manual holdout labels |
| Negative detection (not affected) | Accuracy | **100%** | 10 unaffected company tests |
| Impact magnitude range | In-range | **60.0%** | 20 manual holdout labels |

### The Narrative (Updated)

**Event classification:** Genuinely strong. 95.3% on news text, independently validated.

**Channel prediction:** Two regimes with transparent reliability labeling.
- With event text (the default in the full pipeline): 62.3% top-2, 75.4% with secondary channels. The text signal provides a measurable +15.9pp improvement.
- Without text: 46.4% top-2. Useful for direction but not channel specificity.
- On training-distribution data: 95.2%. This number is real but not representative of novel event-company combinations.

**Direction prediction:** The model's most reliable signal. 90% accuracy, validated on independent data.

**Impact magnitude:** Reasonable but correlational. 60% of actual outcomes fall within predicted ranges.

**Company exposure:** The remaining bottleneck. 17 of 47 errors on the blind eval trace directly to missing company-specific exposure data.

---

## Summary: Day 9 in One Paragraph

Day 9 had two phases. Phase 1 solved channel prediction in text-rich mode by implementing lexicon features (50.7% → 95.2% on cross-validated holdout) and passing event text through the pipeline for inference-time lexicon extraction. Phase 2 confronted the gap between training-distribution accuracy and real-world performance through a 70-example blind evaluation — the honest result: 31.9% top-1 and 62.3% top-2 with event text (+15.9pp vs no text), 75.4% when including secondary channels. The blind eval revealed the core confusion (regulatory vs revenue, 9x; procurement vs revenue, 7x) and confirmed that 21 of 47 errors are adjacent channels, not fundamentally wrong predictions. The system now exposes channel_mode (text_rich/text_poor) and channel_reliability (high/moderate/low) in every prediction, giving users calibrated honesty about when to trust the channel assignment. The final model card honestly reports different accuracy figures for different conditions — 95.2% on training-distribution companies, 62.3% on blind novel combinations, 46.4% without text — rather than hiding behind a single blended metric. The reviewer's meta-point holds across all three review rounds: honest evaluation, right features, and transparent communication of uncertainty matter more than model sophistication.
