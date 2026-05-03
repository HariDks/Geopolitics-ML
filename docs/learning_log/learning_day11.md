# Learning Day 11 — The 4-Week Improvement Sprint

**Project:** Geopolitical Muscle ML Model  
**Date:** May 2-3, 2026  
**Written for:** Understanding the systematic improvement of the ML system across 11 tasks — from honest evaluation foundations through new data sources to embedding-based classifiers and weak supervision

---

## Where We Left Off

At the end of Day 10 we had:
- A working dashboard deployed on GitHub and Streamlit Cloud
- Three rounds of product review incorporated
- Sign correction, severity separation, and 8-section result layout
- Honest model card: 64% classifier on unseen events, 62% channel top-2, 43% impact coverage
- All code public at github.com/HariDks/Geopolitics-ML

Then we received a comprehensive improvement plan (9 sections, 8 major fixes) from a technical reviewer. Day 11 was about executing the plan — 4 weeks compressed into focused implementation.

---

## Part 1: The Improvement Plan

The plan identified the gap between "good personal project" and "publishable, defensible tool." Eight fixes were prioritized:

1. Make the classifier honest about confidence
2. Rebuild the impact predictor with real understanding
3. Use proper statistical methods (conformal prediction)
4. Upgrade strategies with retrieval (RAG)
5. Add a learning feedback loop
6. Get much more training data
7. Add rigorous validation and benchmarking
8. Make it reproducible and deployable

We organized these into 4 weeks, explicitly deferring some items (hierarchical Bayesian models, full causal inference, Docker containerization) to Month 2-3.

We also pushed back on two recommendations:
- "Delete lexicon features entirely" — the lexicon IS the signal (95.2% cross-company validated), the problem is coverage not approach
- "LLM layer for every model" — expensive per call, adds latency, use for strategy recommender only

---

## Part 2: Week 1 — Honest Foundations

### Task 1.1: Time-Based Train/Test Splits

**The problem:** Random train/test splits leak future information. An event from 2024 can appear in training while a 2022 event is in the test set. This inflates accuracy.

**What we did:** Split all data temporally:
- Train: events ≤ 2022 (367 labels, 12 events)
- Validation: events from 2023 (75 labels, 5 events)
- Test: events from 2024+ (160 labels, 3 events)

Applied to all three models by:
- Adding `pipelines/temporal_split.py` utility
- Filtering classifier's SQL queries with `AND event_date <= '2022-12-31'`
- Replacing `train_test_split(random)` with temporal index selection in exposure scorer and impact estimator

**Results:**
| Model | Random split | Temporal split | Drop |
|-------|:-----------:|:--------------:|:----:|
| Exposure scorer macro F1 | 0.877 | 0.728 | -0.149 |
| Exposure scorer accuracy | 86.8% | 69.7% | -17.1pp |
| Impact coverage | 80.7% | 76.6% | -4.1pp |

The drop is expected and healthy — these are honest numbers now. The 17.1pp drop in exposure accuracy means 17% of the previously reported accuracy was "free" from future leakage.

### Task 1.2: Conformal Prediction

**The problem:** Our quantile regression claimed 80% coverage but only 43% of actual outcomes fell inside the predicted interval on holdout data.

**What we did:** Replaced quantile regression with MAPIE's CrossConformalRegressor — a mathematical framework that guarantees coverage by measuring the model's actual errors on a calibration set and making intervals wide enough to account for them.

**How it works:** Train a base XGBoost model. Measure how wrong it is on calibration data. Set the interval width to cover 90% of those measured errors. The guarantee is distribution-free — it works regardless of how bad the model is.

**Result:** 93.8% coverage on the temporal validation set (target was 90%). Mathematically guaranteed.

The MAPIE v1.4 API was tricky — the class names changed from `MapieRegressor` to `CrossConformalRegressor`, the method from `fit` to `fit_conformalize`, and the output from `predict_set` to `predict_interval` returning a tuple. Three iterations before it worked.

### Task 1.3: Feedback Loop Database

**The problem:** User feedback went into a flat CSV with no link to predictions. No prediction logging. No path to retraining.

**What we did:** Built three database tables:
- `predictions`: logs every prediction with full inputs, outputs, confidence, mode, model version
- `corrections`: human corrections linked to specific predictions via foreign key
- `model_versions`: tracks retraining runs and accuracy over time

Plus two scripts:
- `pipelines/prediction_logger.py`: log_prediction(), log_correction(), get_stats(), get_corrections_for_retraining()
- `pipelines/retrain.py`: checks correction count, triggers retraining when threshold met

Dashboard updated to log every prediction and link feedback to its prediction_id.

### Task 1.4: Classifier Confidence Calibration

**The problem:** The classifier outputs a confidence number but doesn't flag when it's likely wrong.

**What we did:** Added `confidence_level` field to every classification output:
- `high` (≥70%): model is fairly sure
- `moderate` (40-70%): check the second-best category
- `low` (<40%): essentially guessing — triggers the embedding backup (Task 3.1)

---

## Part 3: Week 2 — Impact Model Rebuild

### Task 2.1: Split Model 3 into 3A and 3B

**The problem:** The old impact model trained on a blend of stock returns and revenue changes. A stock can drop 10% from panic while actual revenue drops 1%. Blending produces nonsense predictions.

**What we did:** Created two separate models:
- **Model 3A (Market Reaction):** Trained on 1,775 event studies. Target: 5-day stock return. Features: event category + prior returns. MAE: 1.59pp. Direction accuracy: 69.9%.
- **Model 3B (Revenue Impact):** Trained on 18 seed labels with actual revenue_delta_pct. Target: quarterly revenue change. Features: event category, channel, sector, geographic concentration, exposure proxies, lexicon scores. MAE: 46.96pp (too few samples for reliable prediction).

3B includes 3A's stock reaction as an INPUT FEATURE (not as the target). The stock reaction is a useful signal for predicting revenue impact — if traders sold hard, revenue probably drops too. But 3B predicts revenue, not stock price.

Model 3B's poor performance (46.96pp MAE on 18 samples) validates the improvement plan's Week 4 recommendation to scale training data through weak supervision.

### Task 2.2: OpenSanctions

**The problem:** We only used OFAC (586 US sanctions entries). Geopolitical sanctions are global — EU, UK, UN, and 200 other jurisdictions all maintain lists.

**What we did:** Built `pipelines/ingest_opensanctions.py` that streams the 460MB OpenSanctions consolidated CSV, filtering to persons and organizations, mapping to our sanctions taxonomy.

**Result:** 2,000 entries stored — 3.4x our previous OFAC coverage. 88% of new entries were from non-US jurisdictions we had zero visibility into before.

The streaming approach (`resp.iter_lines()`) was necessary because the CSV is 460MB — too large for `pd.read_csv()` to load into memory at once. Single TCP connection, no port exhaustion risk.

### Task 2.3: EPU and GPR Macro Indices

**The problem:** No external benchmark to validate our GRI against. No macro-level features for the impact estimator.

**What we did:** Ingested two established academic indices:
- **EPU (Economic Policy Uncertainty):** Baker, Bloom, Davis 2016. 195 monthly values from 2010-2026. Measures newspaper mentions of policy uncertainty.
- **GPR (Geopolitical Risk Index):** Caldara, Iacoviello 2022 (Federal Reserve). 5,965 daily values from 2010-2026. Measures newspaper mentions of geopolitical tensions.

Created a `macro_indices` table in the database. These serve two purposes: (1) validate our GRI by checking correlation, and (2) add macro context features to the impact estimator.

---

## Part 4: Week 3 — Smarter Classification and Grounded Strategies

### Task 3.1: Embedding-Based Backup Classifier

**The problem:** When DistilBERT is uncertain (confidence < 40%), it still forces an answer from the same neural network that's confused. We need a fundamentally different approach for hard cases.

**What we did:** Built a similarity-based fallback using sentence-transformers (all-MiniLM-L6-v2) + ChromaDB:

1. **Build index:** Embed 2,211 labeled events from ACLED, GTA, OFAC, BIS, news augmentation, and seed labels. Store in ChromaDB vector database.

2. **At classification time:** When primary classifier confidence < 40%, embed the new text and find the 5 most similar past events in ChromaDB.

3. **Vote:** If 3+ of 5 neighbors agree on a category (60%+ vote), use their answer. Otherwise keep the uncertain prediction.

The two classifiers work on different principles: DistilBERT learns text patterns ("text starting with 'On [date], armed forces...' = conflict"), while embeddings measure meaning similarity ("this text is semantically close to 5 past conflict descriptions"). When one fails, the other often succeeds.

Test results on news text:
- Tariffs → trade_policy (100% vote, all 5 neighbors agree)
- OPEC → resource_energy (100%)
- Gallium controls → technology_controls (80%)
- Myanmar coup → armed_conflict (60%)
- Ransomware → armed_conflict (40% — uncertain, neighbors split)

### Task 3.2: RAG Strategy Recommender

**The problem:** Strategy recommendations were generic ("route diversification planning") with no real-world grounding. Users couldn't tell if anyone had actually done this.

**What we did:** Built a Retrieval-Augmented Generation system:

1. **Collected 21 historical case studies:** BP Rosneft exit, Shell Russia exit, McDonald's Russia sale, Maersk Red Sea rerouting, NVIDIA compliance chips, H&M vs Nike Xinjiang responses, Apple India diversification, EPAM employee relocation, Siemens "Value at Stake", Colonial Pipeline ransom, Merck insurance lawsuit, etc.

2. **Embedded and indexed** all cases in ChromaDB using sentence-transformers.

3. **At recommendation time:** Embed the user's event + company query, retrieve the 5 most similar historical cases, generate recommendations using template-based synthesis that cites real company names, years, actions, and outcomes.

New output:
```
#1 Evaluate alternative routing or supply paths.
   Maersk (2024) rerouted via Cape of Good Hope during Red Sea attacks.
   Outcome: Rate surge offset higher costs — counterintuitively positive.
   Relevance: 0.66
```

vs old output:
```
#1 Route diversification planning
```

The template approach requires no LLM API calls (zero cost, sub-second latency). LLM synthesis can be added later for richer narrative while still requiring citations from retrieved cases only.

---

## Part 5: Week 4 — Scaling Up Data

### Task 4.1: Weak Supervision with Snorkel

**The problem:** 163 human labels is too few. We need thousands without thousands of hours of manual work.

**What we did:** Implemented Snorkel (Stanford's weak supervision framework) with 15 labeling functions across four signal types:

- **Stock reaction rules (2):** Large positive/negative CAR → revenue_market_access
- **Event category rules (7):** Conflict → logistics, sanctions → treasury, trade → procurement, etc.
- **Text-based rules (4):** EDGAR mention keywords matching channel lexicons
- **Financial signal rules (1):** Revenue drop + stock drop → revenue_market_access

Each function returns either a channel index or ABSTAIN. Snorkel's label model learns which functions are reliable and aggregates their noisy votes into probabilistic labels.

**Result:** 74 confident labels from 1,523 candidates. Lower than the 5,000 target because:
- Most labeling functions have low coverage (stock reaction rules only fire on 8 candidates)
- Snorkel needs 2-3 agreeing votes to be confident, but most candidates get only 1 vote
- The candidate pool (99 companies × 20 events) is too uniform

The labels that passed are reasonable: Amazon + tariffs → procurement (79%), Costco + tariffs → procurement (79%), GE + energy crisis → procurement (82%).

### Task 4.2: Company Universe Expansion

**The problem:** Only 99 S&P 500 companies in the database. Misses mid-caps and internationally-exposed companies that are most affected by geopolitical events.

**What we did:** Curated 153 additional tickers across 14 geopolitically-relevant sectors:
- Defense/Aerospace (10): HII, LHX, TDG, etc.
- Energy (20): OXY, DVN, MPC, HAL, etc.
- Shipping/Logistics (9): EXPD, XPO, ZIM, etc.
- Semiconductors (14): MRVL, ON, SWKS, ADI, NXPI, etc.
- Mining/Materials (15): FCX, NEM, AA, MP, ALB, etc.
- Consumer brands (14): SBUX, YUM, EL, LEVI, etc.
- Banks/Financial (15): WFC, BAC, C, AIG, etc.
- Plus pharma, tech/cloud, retail, telecom, utilities, food, real estate

Total: 253 companies (up from 100).

Built `pipelines/expand_universe.py` with the event study runner infrastructure. Running 153 × 20 = 3,060 event studies is ready to execute — it's rate-limited by yfinance so takes 30-60 minutes.

After expansion, re-running Snorkel should produce significantly more labels because there are 3× more candidate pairs for the labeling functions to vote on.

---

## Part 6: What's Different Now vs Day 10

| Component | Day 10 | Day 11 | Change |
|-----------|--------|--------|--------|
| Train/test splits | Random | **Temporal** (no future leakage) |
| Impact intervals | 76.6% coverage (quantile) | **93.8% guaranteed** (conformal) |
| Prediction logging | CSV file | **Database** (predictions + corrections + versions) |
| Classifier fallback | None | **Embedding backup** (2,211 events, k-NN) |
| Strategy recommendations | Generic ("route diversification") | **Cited precedents** ("Maersk 2024, rerouted via Cape") |
| Impact model | Single blended model | **3A (market) + 3B (revenue)** separated |
| Sanctions data | 586 OFAC entries | **2,586** (+OpenSanctions) |
| Macro benchmarks | None | **EPU + GPR** (6,160 records) |
| Weak supervision | None | **Snorkel** (15 LFs, 74 confident labels) |
| Company universe | 100 S&P tickers | **253 tickers** (infrastructure ready) |
| Confidence calibration | Raw number | **high/moderate/low** labels |

---

## Part 7: What the Improvement Plan Got Right

1. **"Switch all splits to time-based."** This was the single most important methodological fix. 17 percentage points of exposure accuracy were fake.

2. **"Use Conformal Prediction instead of quantile regression."** Three lines of MAPIE code replaced a broken interval approach with a mathematically guaranteed one. 93.8% vs 76.6%.

3. **"Separate stock reaction from revenue impact."** The sign confusion (ransomware → positive prediction) was caused by blending stock reactions with revenue changes. Separating them eliminates the root cause.

4. **"Add a retrieval system for strategies."** The RAG recommender transformed generic strategy names into grounded recommendations with real company precedents. This is the most visible improvement for users.

5. **"Get more training data from reliable open sources."** OpenSanctions tripled our sanctions coverage. EPU/GPR provide academic validation benchmarks. Both took less than a day.

## Part 8: Where the Improvement Plan's Targets Were Missed

1. **Weak supervision produced 74 labels, not 5,000.** The candidate pool was too small (99 companies × 20 events). The expansion to 253 companies should fix this, but the full event studies haven't run yet.

2. **Model 3B (revenue impact) has only 18 training samples after temporal filtering.** 43 revenue labels total, but 25 are from 2023+ (test set). The conformal intervals for 3B are too wide to be useful. This needs the scaled-up labels from weak supervision.

3. **No causal validation yet.** Difference-in-Differences and Synthetic Control were deferred to Month 2. These are the methods that would make the paper publishable in top journals.

---

## Part 9: Files Created/Modified

```
New files:
  pipelines/temporal_split.py          — time-based train/val/test splitting
  pipelines/prediction_logger.py       — prediction + correction database logging
  pipelines/retrain.py                 — scheduled retraining with correction data
  pipelines/ingest_opensanctions.py    — OpenSanctions consolidated sanctions
  pipelines/ingest_macro_indices.py    — EPU + GPR academic indices
  pipelines/weak_supervision.py        — Snorkel labeling functions + aggregation
  pipelines/expand_universe.py         — 153 additional tickers + event study runner
  models/event_classifier/embedding_backup.py — sentence-transformers + ChromaDB fallback
  models/impact_estimator/train_split.py      — separate 3A (market) and 3B (revenue)
  models/strategy_recommender/case_studies.json — 21 historical case studies
  models/strategy_recommender/rag_recommend.py  — RAG retrieval + template synthesis
  data/seed_labels/weak_labels.csv     — 74 Snorkel-generated weak labels

Modified files:
  database/schema.sql                  — predictions, corrections, model_versions tables
  models/event_classifier/train.py     — temporal date filters on all SQL queries
  models/event_classifier/predict.py   — confidence_level + embedding fallback
  models/exposure_scorer/train.py      — temporal split replacing random split
  models/impact_estimator/train.py     — temporal split + conformal prediction wrapper
  dashboard/app.py                     — prediction logging + correction database link
```

---

## Summary: Day 11 in One Paragraph

Day 11 executed the 4-week improvement plan in a single focused session, implementing 11 tasks across foundations, model rebuild, smarter classification, and data scaling. The most impactful changes were: temporal train/test splits revealing that 17 percentage points of exposure accuracy were inflated by future leakage (0.877 → 0.728 honest F1); conformal prediction replacing broken quantile regression with mathematically guaranteed 93.8% interval coverage; splitting the impact model into separate market reaction (3A, 1.59pp MAE) and revenue impact (3B, needs more data) models; adding an embedding-based backup classifier that finds 5 similar historical events when DistilBERT is uncertain; building a RAG strategy recommender that cites real company precedents ("Maersk rerouted via Cape in 2024"); ingesting OpenSanctions (3.4× sanctions coverage), EPU, and GPR indices; implementing Snorkel weak supervision (74 confident labels from 15 labeling functions); and expanding the company universe from 100 to 253 tickers. The infrastructure is built for all improvement plan items; the main gap is running the expanded event studies (3,060 yfinance calls) and re-running Snorkel on the larger candidate pool to reach the 5,000-label target.
