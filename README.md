# Geopolitical Impact Tester

An ML system that analyzes how geopolitical events affect specific companies — identifying the business mechanism (supply chain, revenue, assets, reputation), estimating financial impact, and explaining its reasoning.

Built on the WEF/IMD/BCG *Building Geopolitical Muscle* framework (2026).

## What it does

Input an event description and a company. The system:

1. **Classifies** the event into 8 geopolitical categories
2. **Identifies** which part of the business is affected (10 impact channels)
3. **Estimates** financial impact range (% of revenue + dollar estimate)
4. **Explains** which text signals drove the prediction
5. **Labels** its own confidence (text-rich vs text-poor mode)

## Honest performance

These are out-of-sample numbers on data the model has never seen during training:

| Capability | Accuracy | How measured |
|------------|:--------:|:------------:|
| Direction (help or hurt?) | **86-90%** | Validated across all evaluations |
| Channel prediction (with text) | **62% top-2** | 70 blind eval pairs |
| Channel prediction (without text) | **46% top-2** | Same 70 blind eval pairs |
| Negative detection (unaffected) | **100%** | 10 unrelated company tests |
| Impact interval coverage | **43%** | Temporal holdout |

The model is strongest at **direction** and **mechanism identification**. It is weakest at **impact magnitude**, particularly for companies with concentrated geographic exposure.

Full evaluation details: [`backtest/RESULTS.md`](backtest/RESULTS.md)

## What powers it

- **7.76M geopolitical events** from GDELT, ACLED, Global Trade Alert, OFAC, BIS, SEC EDGAR
- **163 gold-labeled** company-event impact pairs (human-verified)
- **439 weakly-supervised** labels (auto-generated with documented biases)
- **4 ML models**: DistilBERT classifier, XGBoost exposure scorer, quantile regression impact estimator, retrieval-based strategy recommender

## Try it

**Dashboard:** `streamlit run dashboard/app.py`

**API:** `uvicorn api.app:app --port 8000`

**CLI:**
```bash
python models/pipeline.py "Russia invaded Ukraine triggering sanctions" --ticker AAPL --revenue 383000000000
```

## Project structure

```
dashboard/          — Streamlit web app (3 pages: overview, preloaded examples, custom analysis)
api/                — FastAPI REST endpoints
models/             — 4 trained models + pipeline
  event_classifier/ — DistilBERT (ONNX) for event classification
  exposure_scorer/  — XGBoost for channel prediction + lexicon features
  impact_estimator/ — XGBoost quantile regression for financial impact
  strategy_recommender/ — Retrieval-based with 148 curated strategies
pipelines/          — Data ingestion (6 sources) + data prep + auto-labeling
backtest/           — 10-event backtest + 37x10 risk matrix + blind evaluation
index/              — Geopolitical Risk Index (daily 0-100 composite score)
data/               — Seed labels, mappings, exposure proxies
docs/               — Learning log, technical reference, review response
```

## Key findings

From the [S&P 500 Geopolitical Risk Matrix](backtest/FINDINGS.md) (370 company-event analyses):

- **81% of companies have an unexpected #1 geopolitical risk** — second-order effects dominate
- **US-China tariffs hurt 100% of companies analyzed** — the only truly systemic scenario
- **Same-sector companies diverge by up to 34%** on the same event
- **EU regulation scores higher than any war** in average severity

## Known limitations

- Impact estimates are **correlational, not causal** — based on historical patterns
- **No firm-specific exposure data** (revenue by geography, supplier network) — this is the primary source of prediction errors
- Channel prediction accuracy drops significantly without descriptive event text
- Auto-generated training labels have documented GICS sector bias ([docs/review_response.md](docs/review_response.md))

## Development log

The project's development process is documented transparently across 10 learning days:

[`docs/learning_log/`](docs/learning_log/) — from initial data pipeline through model training, review feedback, and dashboard iteration.

## Roadmap

See [`ROADMAP.md`](ROADMAP.md) for the 4-week improvement plan including temporal splits, conformal prediction, embedding-based classification, and RAG strategy recommendations.

## License

MIT License. See [`LICENSE`](LICENSE).

This tool is designed to explore patterns in how geopolitical events affect companies. **It is not financial advice.**
