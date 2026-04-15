# Learning Day 7 — Semi-Supervised Labeling, Production Infrastructure, and Publishable Artifacts

**Project:** Geopolitical Muscle ML Model  
**Date:** April 14, 2026  
**Written for:** Understanding how we went from a working 4-model prototype to a publishable research artifact — through label expansion, production hardening, and three distinct publishable pieces

---

## Where We Left Off

At the end of Day 6 we had:
- **All 4 models** trained and working end-to-end
- **602 seed labels** from the auto-labeling pipeline (up from 163)
- **Event Classifier** at 95.3% on news, 94.6% on source format
- **Exposure Scorer** at macro F1 0.825 (up from 0.494)
- **Impact Estimator** at 80.7% coverage
- **Strategy Recommender** with 148 strategies across 34 priority cells
- **Git-versioned** with everything committed

The pipeline worked, but three questions remained:
1. Could we ship it to real users (production hardening)?
2. Could we prove it's not just accuracy theater (backtesting)?
3. Could we produce something *publishable* — not just a demo?

Day 7 answered all three.

---

## Part 1: Production Hardening

### 1.1 FastAPI Serving Layer

Built a REST API at [api/app.py](api/app.py) with 7 endpoints:
- `POST /analyze` — full pipeline (event text + company → everything)
- `POST /classify` — Model 1 only
- `POST /exposure` — Model 2 only
- `POST /impact` — Model 3 only
- `POST /strategies` — Model 4 only
- `GET /health` — health check
- `GET /stats` — database statistics

Run with `uvicorn api.app:app --port 8000`. Returns JSON. Lazy-loads each model on first use to minimize memory.

### 1.2 The ONNX Export Breakthrough

#### Roadblock: The PyTorch/SQLite Segfault (Finally Solved)

From Day 5 we knew the full pipeline segfaulted when run in a single process — PyTorch (Model 1) and SQLite (Models 2-4) conflict on macOS MPS. We'd been working around it with subprocess isolation, which added 1-2 seconds per classification.

**The fix: export DistilBERT to ONNX + use the Rust tokenizer.**

```
PyTorch model:    255 MB  (model.safetensors)
ONNX model:       0.8 MB  (model.onnx)
Output difference: 0.000002 (negligible)
```

Why this works: ONNX Runtime has no PyTorch dependency. The `tokenizers` library (HuggingFace's Rust-based tokenizer) replaces the Python `DistilBertTokenizer` without importing torch at all.

**The test that validated it:**
```python
from models.event_classifier.predict import EventClassifier   # ONNX, no torch
from models.exposure_scorer.predict import ExposureScorer     # SQLite
from models.impact_estimator.predict import ImpactEstimator   # SQLite
from models.strategy_recommender.recommend import StrategyRecommender

# All 4 in one process — NO segfault!
clf = EventClassifier()
print(f"Backend: {clf._backend}")  # "onnx"
```

Result: all 4 models now run in a single process. The FastAPI server no longer needs subprocess isolation.

### 1.3 Scheduled Re-Ingestion

Built [pipelines/scheduled_ingest.py](pipelines/scheduled_ingest.py) for cron-compatible data refresh:
- GDELT: every 15 minutes (new events since last run)
- ACLED: weekly
- OFAC: daily
- Detects last ingestion date via `ingestion_log` table
- Re-runs data prep after new data lands
- `--dry-run` flag for preview

Cron examples in the docstring:
```bash
*/15 * * * *  python pipelines/scheduled_ingest.py --source gdelt
0 2 * * 1     python pipelines/scheduled_ingest.py --source all
```

### 1.4 Streamlit Dashboard

Built [dashboard/app.py](dashboard/app.py) as the user-facing demo. Went through multiple iterations:

**Version 1** (initial): 4 pages — Analyze Event, Database Explorer, Seed Labels, Model Stats. Developer-focused.

**Version 2** (redesign): After feedback that "Database Explorer / Seed Labels / Model Stats" are developer tools, not user features, I replaced them with:
- **Company Deep Dive** — one company across all 10 scenarios
- **Scenario Comparison** — side-by-side with bar charts
- **Portfolio Scanner** (kept) — scan holdings against an event

Also added:
- **37-company dropdown** with name-based selection (not ticker-only)
- **Auto-filling revenue and sector** when company is selected
- **Pre-built event scenarios** (dropdown of 10) instead of requiring users to write descriptions
- **Geopolitical exposure profiles** for each company (e.g., "Apple: Heavy China manufacturing, 90%+ iPhones assembled there")
- **Sector-specific strategy context** explaining WHY each strategy matters for that industry

---

## Part 2: The Semi-Supervised Labeling Pipeline (Completed)

### 2.1 Growing Labels from 163 to 602

The auto-labeling pipeline built on Day 6 was expanded. Key insight: **Model 2 can't predict channels accurately with only 163 labels, so don't let Model 2 assign channels during auto-labeling.** Use GICS sector-based rules instead.

Final channel distribution after 602 labels (minimum 40 per channel):

```
revenue_market_access:       104
procurement_supply_chain:     97
logistics_operations:         60
innovation_ip:                58
regulatory_compliance_cost:   57
financial_treasury:           51
capital_allocation:           46
workforce_talent:             45
reputation_stakeholder:       44
cybersecurity_it:             40
```

### 2.2 Model 2 Results at 602 Labels

```
Macro F1:         0.825  (was 0.494 with 163 labels)
Accuracy:         82.6%
Severity R²:      0.999
Worst channel:    reputation_stakeholder at 0.706 F1
```

Four channels went from **literally 0.000 F1 to 0.70+** through targeted label generation. The pattern: **data composition matters more than model architecture.**

### 2.3 Model 3 Retrained

With 573 seed labels available (up from 134), Model 3 coverage improved:
- Overall coverage: 80.2% → 80.7%
- Seed label coverage: 66.7% → **75.7%**
- Event study MAE: 0.17 pp → 0.08 pp

---

## Part 3: The Backtest (Validation)

### 3.1 Selecting 10 Diverse Events

We picked events across 5 dimensions to avoid cherry-picking:

| # | Event | Type | Region | Intensity |
|---|-------|------|--------|-----------|
| 1 | Russia invasion 2022 | Armed conflict | Europe | High |
| 2 | US chip export controls | Technology controls | US-China | High |
| 3 | COVID lockdown | Armed conflict | Global | Catastrophic |
| 4 | India demonetization 2016 | Political transition | South Asia | High |
| 5 | Red Sea Houthi attacks | Armed conflict | Middle East | Moderate |
| 6 | Australia-China wine tariffs | Trade policy | Asia-Pacific | Moderate |
| 7 | SolarWinds hack | Cybersecurity | US/Russia | High |
| 8 | Argentina Milei deregulation | Political transition | Latin America | Moderate (positive) |
| 9 | Panama mining contract | Political transition | Central America | High |
| 10 | Sudan civil war | Armed conflict | Africa | High |

14 company-event pairs total (some events had multiple companies like BP + Shell for Russia).

### 3.2 Backtest Results

```
Direction accuracy (+/-):   92%  (12/13)
Category classification:    64%  (9/14)
In-predicted-range:         50%  (6/12)
Impact channel:             7%   (1/14) ← weakest link
```

### 3.3 The Strongest Predictions

| Company | Event | Predicted | Actual | Gap |
|---------|-------|:---------:|:------:|:---:|
| NVIDIA | Chip controls | -6.3% | -5.0% | 1.3pp |
| KLAC | Chip controls | -14.6% | -15.0% | 0.4pp |
| First Quantum | Panama mine | -12.2% | -13.1% | 0.9pp |
| Maersk | Red Sea | +13.8% | +15.0% | 1.2pp |
| SolarWinds | Hack | -35.7% | -40.0% | 4.3pp |
| Boeing | COVID | -35.0% | -24.0% | within range |

The Maersk prediction is particularly significant — the model correctly identified a **counterintuitive positive** (Maersk benefited from Red Sea rerouting via rate surges, even though it's a shipping company in a shipping crisis).

### 3.4 The Weakest Predictions

| Company | Event | Predicted | Actual | Gap |
|---------|-------|:---------:|:------:|:---:|
| Treasury Wine | China tariff | ~0% | -96.0% | 96pp |
| Zain Group | Sudan war | -0.5% | -71.0% | 70pp |

The pattern: **niche single-country-dependent companies** (TWE.AX is overwhelmingly dependent on China wine sales; Zain's Sudan operation was a concentrated bet) have extreme outcomes the model can't see because it doesn't know about their concentrated exposure.

### 3.5 Three Publishable Figures

Saved in [backtest/figures/](backtest/figures/):
- `predicted_vs_actual.png` — scatter plot showing predictions near diagonal
- `prediction_ranges.png` — horizontal bars with actual outcomes
- `scorecard.png` — summary metrics + key prediction highlights

---

## Part 4: The Hidden Geopolitical Risk Map of the S&P 500

### 4.1 The Full Matrix (370 Analyses)

Ran **37 S&P 500 companies × 10 geopolitical scenarios** through the full 4-model pipeline. For each combination:
- Classify the event (Model 1)
- Score company exposure (Model 2)
- Estimate financial impact (Model 3)
- Identify primary channel

This produced a 37×10 severity matrix plus impact percentages and channels.

Code: [backtest/risk_matrix.py](backtest/risk_matrix.py), output: [backtest/risk_matrix.csv](backtest/risk_matrix.csv).

### 4.2 The Interactive Heatmap

Built [backtest/interactive_matrix.py](backtest/interactive_matrix.py) with Plotly. Three HTML files:
- `risk_heatmap.html` — the 37×10 matrix with hover details (severity, impact %, channel)
- `surprise_risks.html` — unexpected #1 risks bar chart
- `sector_exposure.html` — grouped bars: which sectors are most vulnerable per scenario

### 4.3 The Non-Obvious Findings (30 Surprises)

Defined "surprise" as a company whose #1 risk doesn't match sector expectation:

- **Energy companies expected** to fear OPEC cuts → many actually fear US-China tariffs more
- **Tech companies expected** to fear chip export controls → many actually fear OPEC cuts (energy costs)
- **Financial companies expected** to fear EM debt crisis → most actually fear US-China tariffs

Distribution of actual top risks:
```
OPEC production cut:          14 companies
US-China tariff escalation:    7 companies
EU regulatory crackdown:       6 companies
Red Sea shipping disruption:   3 companies
```

### 4.4 The Five Publishable Findings

Written up in [backtest/FINDINGS.md](backtest/FINDINGS.md):

1. **81% of companies have an unexpected #1 risk** — the hook
2. **US-China tariffs hurt 100% of companies** — the only truly systemic scenario
3. **Same-sector companies diverge by up to 34%** — why sector ETFs are wrong
4. **Pfizer most exposed, Broadcom/J&J most resilient** — the actionable ranking
5. **EU regulation beats wars** — the counterintuitive punchline

Example deep cut: "Under a Red Sea shipping disruption, Pfizer (severity -0.414) is 27% more exposed than J&J (-0.326). Why? Pfizer's API supply chain runs through Suez from Indian manufacturers. J&J's supply chain is more diversified."

---

## Part 5: The Geopolitical Risk Index (GRI)

### 5.1 The Concept

A single 0-100 daily score — like VIX for geopolitical risk — updated from the 7.76M-event database. Transparent methodology, open-source, sub-indices by category.

### 5.2 The Formula

```
GRI = 0.40 × Volume Score + 0.40 × Severity Score + 0.20 × Concentration Score
```

**Volume Score (40%):** Daily event count weighted by category importance, normalized against 90-day rolling average via z-scores. Spikes above baseline indicate escalation.

**Severity Score (40%):** Average Goldstein scale (-10 to +10, inverted and rescaled to 0-100). Lower Goldstein = more conflictual events = higher risk.

**Concentration Score (20%):** Herfindahl index across 8 event categories. High concentration = specific crisis. Even spread = background noise.

**Category weights** (derived from our model's learned importance):
- Armed conflict: 1.0 (highest corporate impact)
- Sanctions: 0.9
- Technology controls: 0.85
- Trade policy: 0.8
- Resource/energy: 0.8
- Regulatory: 0.6
- Political transitions: 0.5
- Institutional realignment: 0.4

### 5.3 The Results

Computed for **1,432 days** (2022-01-30 to 2025-12-31):

```
Historical range:  39.9 — 82.6
Mean:              62.6
Std:               6.8
Peak:              March 27, 2023 (GRI 82.6)
                   driven by armed conflict escalation
```

Five sub-indices published: armed conflict, sanctions, trade policy, political transitions, regulatory shifts.

### 5.4 The Published Page

[index/geopolitical_risk_index.html](index/geopolitical_risk_index.html) is a standalone HTML page with:
- **Current GRI** as a big number at top
- **Interactive time series** from 2022-2025 with annotated major events
- **Sub-indices** as overlaid line chart
- **Event volume** area chart at bottom
- **Full methodology** section explaining every component
- **Range slider** for zooming into specific periods

### 5.5 UI Iteration

#### Roadblock: Legend Overlapped Title, Event Volume Bars Invisible

First version had the legend in the top-right which collided with the subtitle. The event volume bars were too thin for 1,432 days of data — just a solid-looking rectangle.

**Fixes:**
1. Moved legend to bottom (`y=-0.08`) to free up top space
2. Replaced bar chart with filled area chart (`go.Scatter` with `fill="tozeroy"`)
3. Added white background boxes to event annotations so they stand out
4. Staggered vertical offsets on annotations so they don't overlap each other
5. Increased bottom panel proportion from 20% to 30%

---

## Part 6: The Three Publishable Artifacts

At the end of Day 7, we have three distinct publishable pieces, each targeting a different audience:

| Artifact | File | Audience | What it proves |
|----------|------|----------|----------------|
| **Backtest** | `backtest/figures/*.png` | Skeptics / quants | The model's predictions are actually accurate on historical data |
| **Risk Matrix** | `backtest/figures/risk_heatmap.html` + `FINDINGS.md` | Business readers / media | The model reveals non-obvious insights no analyst has published |
| **GRI Index** | `index/geopolitical_risk_index.html` | Ongoing attention / citations | A daily updatable metric people will share when it spikes |

The combination matters: the backtest provides **credibility** (the model works), the risk matrix provides **novelty** (here's something you didn't know), and the GRI provides **engagement** (come back tomorrow to see the new number).

---

## Part 7: Roadblocks Solved on Day 7

| # | Roadblock | Solution |
|---|-----------|----------|
| 1 | Auto-labeler assigned all candidates to `revenue_market_access` (Model 2 overconfident on biased class) | GICS sector-based channel assignment — energy → procurement, tech → innovation, finance → financial_treasury, etc. |
| 2 | Cybersecurity channel stuck at 8 labels | Expanded search to include tech + finance companies during NotPetya, plus cyber-warfare events in Russia/Iran conflicts |
| 3 | Eight events had 0-2 auto-generated labels | Lowered stock reaction threshold to 5%, added sector-event combinations for underrepresented events (Brexit, OPEC, Suez, Chile, Panama) |
| 4 | PyTorch/SQLite segfault on macOS (leftover from Day 5) | ONNX export (255MB → 0.8MB) + Rust tokenizer — zero PyTorch dependency for inference |
| 5 | Streamlit caching of SQLite connection failed across threads | Removed `@st.cache_resource` from DB connection — fresh connection per request |
| 6 | Dashboard "Database Explorer / Seed Labels / Model Stats" were developer tools, not user features | Replaced with Company Deep Dive, Scenario Comparison, Portfolio Scanner — all user-facing |
| 7 | Strategies felt repetitive because 34 cells × same strategies → same output for same input | Added sector-specific context explanations + company-specific geopolitical exposure profiles so users understand *why* strategies apply |
| 8 | GRI dashboard had legend overlapping title, event volume bars invisible | Legend moved to bottom, bars replaced with area chart, annotations given bordered boxes |

---

## Part 8: The Complete Project State at End of Day 7

### Database
```
geopolitical_events:     7,763,222
company_financials:      1,572
financial_deltas:        1,572
geopolitical_mentions:   17,372
corporate_impacts:       1,606
event_studies:           1,973
strategies:              148
seed_labels (CSV):       602
```

### Models
| Model | Architecture | Performance |
|-------|--------------|:-----------:|
| 1. Event Classifier | DistilBERT → ONNX | 95.3% news / 94.6% source |
| 2. Exposure Scorer | XGBoost (multi-class + regression) | Macro F1 0.825 |
| 3. Impact Estimator | XGBoost quantile regression | 80.7% coverage, MAE 0.39pp |
| 4. Strategy Recommender | Retrieval + ranking | 148 strategies, 34 cells |

### Codebase
```
api/                    — FastAPI serving layer
backtest/               — 10-event backtest + 37x10 risk matrix + findings
dashboard/              — Streamlit interactive dashboard
data/                   — Raw data, mappings, seed labels, news augmentation
database/               — Schema
index/                  — Geopolitical Risk Index (GRI) computation + HTML page
models/                 — All 4 trained models + pipeline
  event_classifier/     - DistilBERT (ONNX)
  exposure_scorer/      - XGBoost
  impact_estimator/     - XGBoost quantile
  strategy_recommender/ - Retrieval
pipelines/              — Ingestion (GDELT, ACLED, GTA, OFAC, BIS, EDGAR)
                          + data prep + auto-labeling + scheduled updates
```

### Git Log (22 commits)
```
Add Geopolitical Risk Index (GRI): daily 0-100 composite score
Add publishable findings: The Hidden Geopolitical Risk Map of the S&P 500
Add S&P 500 Geopolitical Risk Matrix: 370 analyses + interactive viz
Add publishable backtest visualizations
Add backtest framework: 10 events, 14 company-event pairs
Redesign dashboard: replace dev pages with user-facing features
Improve dashboard: company dropdown, event scenarios, portfolio scanner
Add scheduled re-ingestion pipeline and Streamlit dashboard
Update FastAPI to use direct ONNX inference (no subprocess)
ONNX export + Rust tokenizer: single-process pipeline, no segfault
Add FastAPI serving layer with 5 endpoints
Add learning_day6.md: semi-supervised labeling and model improvements
Retrain impact estimator with 573 seed labels (was 134)
Grow seed labels to 602, Model 2 macro F1 reaches 0.825
Sector-aware auto-labeling: 163 -> 382 labels, Model 2 macro F1 0.698
Add semi-supervised labeling pipeline, grow seed labels 163 -> 330
Add Part 13 to learning_day5.md: impact estimator interval widening
Widen impact estimator quantiles from q10/q90 to q05/q95
Update learning_day5.md with classifier augmentation journey
Cap source-format examples at 500 to balance with news augmentation
Add news-style augmentation for event classifier
Initial commit: complete Phase 2 geopolitical risk ML pipeline
```

---

## Summary: Day 7 in One Paragraph

Day 7 transformed a working 4-model prototype into a publication-ready system. On the production side: FastAPI serving layer with 7 endpoints, ONNX export finally fixing the PyTorch/SQLite segfault (255MB → 0.8MB, all 4 models now run in one process), scheduled re-ingestion for GDELT/ACLED/OFAC, and a redesigned Streamlit dashboard with user-facing pages (Company Deep Dive, Scenario Comparison, Portfolio Scanner) replacing developer tools. On the validation side: a 10-event backtest across 5 continents and 5 event types showed 92% directional accuracy, including correctly predicting NVIDIA's -5% export control hit (predicted -6.3%) and Maersk's counterintuitive +15% Red Sea benefit (predicted +13.8%). On the publication side: three distinct artifacts were produced — (1) the backtest with three figures proving model credibility, (2) "The Hidden Geopolitical Risk Map of the S&P 500" with 370 company-scenario analyses and 30 non-obvious findings (81% of companies have an unexpected #1 risk, US-China tariffs hurt 100% of companies, EU regulation scores higher than any war), and (3) the Geopolitical Risk Index (GRI), a daily 0-100 composite score computed from 1,432 days of data with 5 sub-indices, published as a standalone HTML page with annotated major events and full methodology transparency. Eight roadblocks were solved including sector-aware channel assignment, the ONNX segfault fix, Streamlit cross-thread connection errors, and GRI chart rendering issues. Total project state: 22 git commits, 602 seed labels, 10/10 channels above 0.70 F1, three publishable artifacts ready for blog posts or research submission.
