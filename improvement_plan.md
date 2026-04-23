# Geopolitics ML: Implementation Plan

**Based on:** Full Improvement Roadmap (external review document)  
**Date:** April 23, 2026  
**Scope:** 4-week sprint to take the project from "good personal project" to "publishable and defensible"

---

## Current State (Honest Assessment)

| Component | Claimed | Actual (blind eval) | Gap |
|-----------|:-------:|:-------------------:|:---:|
| Event classifier | 95% | 64% on unseen events | Training format bias |
| Channel prediction | 82.5% | 50.7% without text, 62.3% with text | Feedback loop inflation |
| Impact intervals | 80% coverage | 43% on holdout | Quantile approach miscalibrated |
| Direction accuracy | 90% | 86-90% | Genuinely strong |
| Training data | 602 labels | 163 human-verified | 439 are auto-generated with known bias |

These are the numbers we need to improve. Not the reported numbers — the real ones.

---

## What We're NOT Doing (and Why)

| Suggestion from review | Why we're deferring |
|------------------------|---------------------|
| Delete lexicon features entirely | Lexicon IS the signal — 95.2% cross-company validated. Problem is coverage, not approach. Keep lexicon AND add embeddings. |
| Hierarchical Bayesian with PyMC | Significant complexity for marginal gain. Conformal Prediction alone fixes the interval problem with 3 lines of code. |
| Full causal inference (DiD, Synthetic Control) | Publishable but 2-3 weeks each. Month 2 material. |
| LLM layer for every model | Expensive per call, adds latency. Use for strategy recommender only. |
| Expand to Russell 3000 immediately | Infrastructure works but need the foundation fixes first. Week 4. |

These are deferred, not rejected. They're Month 2 and Month 3 items.

---

## Week 1: Honest Foundations

**Goal:** Fix the two most damaging methodological problems — train/test leakage and miscalibrated intervals.

### Task 1.1: Time-Based Train/Test Splits

**What's wrong:** Current splits are random, which leaks future information. An event from 2024 can appear in training while a 2022 event is in the test set. This inflates accuracy.

**What to do:**
- Train on events/labels from 2015-2022
- Validate on 2023
- Test on 2024-2026
- Apply to ALL models (classifier, exposure scorer, impact estimator)
- Re-run all evaluations with temporal splits
- Document the new (likely lower) accuracy numbers honestly

**Files to change:**
- `models/event_classifier/train.py` — add `--temporal-split` flag
- `models/exposure_scorer/train.py` — split seed labels by event date
- `models/impact_estimator/train.py` — same temporal split
- `backtest/holdout_eval.py` — update to use temporal holdout

**Success criterion:** All reported accuracy numbers come from temporal holdout. No future leakage.

**Estimated time:** 1 day

### Task 1.2: Conformal Prediction for Guaranteed Intervals

**What's wrong:** Current quantile regression claims 80% coverage but delivers 43% on holdout. The intervals are not calibrated.

**What to do:**
- Install `MAPIE` library
- Wrap the existing XGBoost impact model with `MapieRegressor`
- Use a calibration set (temporal: 2023 data) to compute conformal intervals
- The output is mathematically guaranteed to cover the specified percentage

**Files to change:**
- `models/impact_estimator/train.py` — add conformal wrapper
- `models/impact_estimator/predict.py` — output conformal intervals
- `requirements.txt` — add `mapie`

**Success criterion:** 90% prediction intervals genuinely cover 90% of outcomes on 2024+ holdout.

**Estimated time:** 1 day

### Task 1.3: Feedback Loop Database

**What's wrong:** User feedback is stored in a flat CSV. No prediction logging, no correction tracking, no retraining pipeline.

**What to do:**
- Create three new database tables: `predictions`, `corrections`, `model_versions`
- Log every prediction with full inputs, outputs, confidence, and model version
- Connect dashboard feedback to the `corrections` table
- Add confidence threshold flagging (below 70% → flagged for review)
- Build a simple retraining script that combines original + corrected labels

**Files to change:**
- `database/schema.sql` — add three tables
- `dashboard/app.py` — log predictions to DB, connect feedback to corrections table
- `pipelines/retrain.py` — new script for scheduled retraining

**Success criterion:** Every prediction is logged. Every correction is linked to its prediction. Retraining script works.

**Estimated time:** 2 days

### Task 1.4: Classifier Confidence Calibration

**What's wrong:** The classifier outputs a confidence score but doesn't know when it's wrong. Low-confidence predictions are just as likely to be shown as high-confidence ones.

**What to do:**
- Add a calibration layer: if confidence < 60%, output "low confidence — flagged for review" instead of a specific category
- Measure calibration: of predictions the model says are 90% confident, are 90% actually correct?
- Use temperature scaling (standard calibration technique) to fix any miscalibration

**Files to change:**
- `models/event_classifier/predict.py` — add confidence threshold
- `models/event_classifier/train.py` — add calibration step after training

**Success criterion:** Calibration plot shows diagonal (predicted confidence ≈ actual accuracy).

**Estimated time:** 0.5 days

---

## Week 2: Impact Model Rebuild

**Goal:** Fix the sign errors and separate stock reaction from revenue impact.

### Task 2.1: Split Model 3 into 3A (Market) and 3B (Revenue)

**What's wrong:** The current model trains on a blend of stock returns and revenue changes. A stock can drop 10% from panic while revenue drops 1%. Blending these produces nonsense.

**What to do:**
- **Model 3A (Market Reaction):** Train only on event study data (stock CAR). Target: 5-day abnormal return. Features: event characteristics, firm size, sector, prior volatility.
- **Model 3B (Revenue Impact):** Train only on seed labels with actual `revenue_delta_pct`. Target: quarterly revenue change. Features: exposure proxies, geographic concentration, channel, event type.
- Each model produces its own prediction. Dashboard shows both:
  - "Short-term market reaction: -3.2%"
  - "Estimated revenue impact: -0.8% to -2.1%"

**Files to change:**
- `models/impact_estimator/train.py` — split into two training paths
- `models/impact_estimator/predict.py` — output both predictions
- `models/impact_estimator/saved/` — separate model files for 3A and 3B
- `dashboard/app.py` — display both in result card

**Success criterion:** No more sign confusion between stock reaction and business impact.

**Estimated time:** 2 days

### Task 2.2: Ingest OpenSanctions

**What's wrong:** We only use OFAC (586 entries). OpenSanctions consolidates OFAC + EU + UK + UN + 200 other watchlists.

**What to do:**
- Download OpenSanctions bulk data (JSON/CSV, free, updated daily)
- Map to our existing sanctions taxonomy
- Store in `geopolitical_events` table with `source = 'opensanctions'`
- ~3x coverage gain over OFAC alone

**Files to create:**
- `pipelines/ingest_opensanctions.py`

**Success criterion:** Sanctions coverage triples. New entries appear in the database.

**Estimated time:** 1 day

### Task 2.3: Ingest EPU and GPR Indices

**What's wrong:** No macro-level benchmarks to validate against or use as features.

**What to do:**
- Download Economic Policy Uncertainty Index (policyuncertainty.com, CSV, free)
- Download Geopolitical Risk Index (matteoiacoviello.com/gpr.htm, CSV, free)
- Store as daily time series in a new `macro_indices` table
- Add as features to the GRI computation (validate our GRI against established indices)
- Add as features to impact estimator (was there elevated macro risk at event time?)

**Files to create:**
- `pipelines/ingest_macro_indices.py`
- `database/schema.sql` — add `macro_indices` table

**Success criterion:** Our GRI correlates meaningfully with the established GPR index.

**Estimated time:** 1 day

---

## Week 3: Smarter Classification and Strategy Grounding

**Goal:** Fix the classifier's generalization gap and make strategy recommendations cite real precedents.

### Task 3.1: Embedding-Based Backup Classifier

**What's wrong:** When DistilBERT is uncertain (confidence < 60%), it still outputs a (likely wrong) category. We need a backup that works differently.

**What to do:**
- Use `sentence-transformers` with `all-MiniLM-L6-v2` (fast, free, runs on CPU)
- Compute embeddings for all labeled events in the database
- Store in ChromaDB (open source vector DB, runs locally)
- When DistilBERT confidence < 60%, find the 5 most similar past events by embedding similarity
- If 4/5 agree on a category, use that. Otherwise flag as uncertain.

**Files to create:**
- `models/event_classifier/embedding_backup.py`
- Store embeddings in `data/processed/event_embeddings.db` (ChromaDB)

**Files to change:**
- `models/event_classifier/predict.py` — add fallback logic

**Success criterion:** Classifier on 2024+ temporal holdout improves from 64% to 75%+. Uncertain cases correctly flagged.

**Estimated time:** 2 days

### Task 3.2: RAG for Strategy Recommender

**What's wrong:** Strategy recommendations are generic ("route diversification planning") with no real-world grounding. Users can't tell if anyone has actually done this.

**What to do:**
- Collect 200-500 historical case studies (WEF report cases, Yale CELI Russia tracker, HBS case summaries, news coverage of Red Sea/chip controls responses)
- Compute embeddings and store in ChromaDB
- When recommending strategies, retrieve the 10-15 most similar historical cases
- Send to Claude/GPT-4 with a structured prompt: "Based on these precedents, recommend 5 strategies with citations"
- Output: grounded recommendations with "Siemens did X in 2019, outcome was Y"

**Files to create:**
- `models/strategy_recommender/case_studies.json` — structured case study database
- `models/strategy_recommender/rag_recommend.py` — retrieval + LLM synthesis
- `pipelines/ingest_case_studies.py` — scraper/parser for case study sources

**Files to change:**
- `dashboard/app.py` — show strategy recommendations with citations

**Success criterion:** Each recommendation cites at least 2 real historical precedents.

**Estimated time:** 3 days

---

## Week 4: Scale Up Data

**Goal:** Go from 602 labels to 5,000+ and from 100 companies to 3,000+.

### Task 4.1: Weak Supervision with Snorkel

**What's wrong:** 163 human labels is too few. 439 auto-labels have known GICS bias. We need thousands of labels without thousands of hours of human work.

**What to do:**
- Install Snorkel (Stanford's weak supervision framework)
- Write 15-20 labeling functions, each imperfect but informative:
  - "If 8-K filed within 30 days of event AND stock dropped >3%, label as impacted"
  - "If 10-K mentions event keywords AND revenue_delta > 5%, label as [channel]"
  - "If company sector matches event type's typical channel, label as [channel]"
  - "If EDGAR mention specificity > 50, label as impacted"
- Snorkel aggregates these noisy rules into probabilistic labels
- Audit 5% by human review to measure quality
- Tag all Snorkel labels as `labeled_by = 'weak_supervision'` — never mix with gold labels

**Files to create:**
- `pipelines/weak_supervision.py` — labeling functions + Snorkel aggregation

**Success criterion:** 5,000+ probabilistic labels with documented 80-85% accuracy.

**Estimated time:** 3 days

### Task 4.2: Expand Company Universe

**What's wrong:** Only 99 S&P 500 companies in the database. Misses mid-caps and international companies that are often most affected by geopolitical events.

**What to do:**
- Extend `ingest_edgar.py` to Russell 3000 (same pipeline, more tickers)
- Extend event studies to cover Russell 3000 (~3,000 companies x 20 events)
- Extend geographic segment extraction to new companies
- Re-run exposure proxy extraction for new companies

**Files to change:**
- `pipelines/ingest_edgar.py` — expand ticker list
- `pipelines/extract_exposure_proxies.py` — run on new companies
- `pipelines/ingest_geo_segments.py` — run on new companies

**Success criterion:** 3,000+ companies with financial data, event studies, and exposure proxies.

**Estimated time:** 2 days (mostly runtime, not coding)

### Task 4.3: Retrain All Models on Expanded Data

**What to do:**
- Retrain classifier with temporal split on expanded data
- Retrain exposure scorer with 5,000+ weak supervision labels
- Retrain impact estimator 3A/3B on expanded event studies
- Re-run conformal prediction calibration
- Re-run all evaluations on temporal holdout
- Document improvements (or lack thereof) honestly

**Success criterion:** At least one metric improves by 5+ percentage points over Week 1 baseline.

**Estimated time:** 1 day

---

## What Comes After (Month 2-3, Not This Sprint)

| Item | Why it's deferred | When |
|------|-------------------|------|
| Causal inference (DiD, Synthetic Control) | 2-3 weeks per method, needs clean identification | Month 2 |
| Hierarchical Bayesian for Model 3B | Conformal prediction may be sufficient | Month 2, if intervals still poor |
| LLM-assisted labeling at scale | Needs cost budget + audit pipeline | Month 2 |
| Full baseline comparison (human analysts, zero-shot LLM) | Needs external participants | Month 2 |
| Docker containerization | Needed for publication, not for development | Month 3 |
| Paper draft | After all validation is complete | Month 3 |
| Public leaderboard | After paper submitted | Month 3 |
| Multi-armed bandit for strategies | Needs 500+ outcome reports (6+ months of usage) | Month 6+ |

---

## Success Metrics After 4 Weeks

| Metric | Current | Target | Method |
|--------|:-------:|:------:|--------|
| Classifier (temporal holdout) | 64% | 75%+ | Embedding backup + more training data |
| Channel prediction (blind eval) | 62.3% top-2 | 70%+ top-2 | More labels + better features |
| Impact interval coverage | 43% | 85%+ | Conformal prediction |
| Direction accuracy | 86-90% | 90%+ | Separate 3A/3B models |
| Training labels | 163 gold + 439 biased | 163 gold + 5,000 weak supervision | Snorkel |
| Company coverage | 99 S&P 500 | 3,000+ Russell 3000 | Pipeline expansion |
| Strategy grounding | 0 citations | 2+ citations per recommendation | RAG with case studies |
| Feedback loop | CSV file | Full DB with retraining | Database + pipeline |

---

## Daily Standup Format

Each day, update this section:

```
Day X:
- Completed: [task]
- Blocked on: [issue]
- Next: [task]
- Accuracy change: [metric before → after]
```

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Conformal prediction intervals too wide to be useful | Users see "-50% to +50%" and ignore | Tune alpha, consider conditional conformal |
| Weak supervision labels worse than expected | 5,000 noisy labels hurt more than help | Audit 5%, keep gold labels as separate eval set |
| RAG strategy recommendations hallucinate | LLM invents precedents that don't exist | Require citations from retrieved docs only, no generation |
| Russell 3000 expansion overwhelms SQLite | Database too large for single-file DB | Consider PostgreSQL migration (was always planned) |
| Temporal split leaves too few training examples pre-2023 | Model underfits | Augment with weak supervision in Week 4 |

---

*This plan is designed to be executed by reading one task at a time. Each task has specific files to change, a clear success criterion, and a time estimate. Start at Task 1.1 and work forward.*
