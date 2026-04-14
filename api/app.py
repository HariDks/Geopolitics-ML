"""
FastAPI serving layer for the Geopolitical Risk Pipeline.

Provides REST endpoints for each model and the full pipeline.
Uses subprocess isolation for Model 1 (DistilBERT) to avoid the
PyTorch/SQLite segfault on macOS.

Usage:
    uvicorn api.app:app --reload --port 8000

Endpoints:
    POST /analyze           — Full pipeline (text + ticker → classification + exposure + impact + strategies)
    POST /classify          — Model 1 only (text → event category)
    POST /exposure          — Model 2 only (event + ticker → channel + severity)
    POST /impact            — Model 3 only (event + channel + ticker → low/mid/high)
    POST /strategies        — Model 4 only (event + channel → ranked strategies)
    GET  /health            — Health check
    GET  /stats             — Database stats
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

app = FastAPI(
    title="Geopolitical Risk API",
    description="ML-powered geopolitical event classification, company exposure scoring, financial impact estimation, and strategy recommendation.",
    version="1.0.0",
)

# ── Request/Response Models ──────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    text: str = Field(..., description="Event text to classify", min_length=5)

class ClassifyResponse(BaseModel):
    category: str
    confidence: float
    all_scores: dict[str, float]

class ExposureRequest(BaseModel):
    event_category: str
    ticker: str = ""
    mention_sentiment: float = 0.0
    car_1_5: float = 0.0

class ExposureResponse(BaseModel):
    channel_prediction: str
    channel_confidence: float
    severity_score: float
    top_3_channels: list[dict]

class ImpactRequest(BaseModel):
    event_category: str
    impact_channel: str = ""
    ticker: str = ""
    mention_sentiment: float = 0.0
    car_1_5: float = 0.0
    revenue_usd: float = 0.0

class ImpactResponse(BaseModel):
    impact_low_pct: float
    impact_mid_pct: float
    impact_high_pct: float
    confidence: float
    impact_low_usd: Optional[float] = None
    impact_mid_usd: Optional[float] = None
    impact_high_usd: Optional[float] = None

class StrategyRequest(BaseModel):
    event_category: str
    impact_channel: str
    severity: float = -0.5
    company_size: str = "large"
    top_n: int = 5

class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Event text to analyze", min_length=5)
    ticker: str = ""
    revenue_usd: float = 0.0
    company_size: str = "large"

# ── Lazy model loading ───────────────────────────────────────────────────────

_scorer = None
_estimator = None
_recommender = None


def get_scorer():
    global _scorer
    if _scorer is None:
        from models.exposure_scorer.predict import ExposureScorer
        _scorer = ExposureScorer()
    return _scorer


def get_estimator():
    global _estimator
    if _estimator is None:
        from models.impact_estimator.predict import ImpactEstimator
        _estimator = ImpactEstimator()
    return _estimator


def get_recommender():
    global _recommender
    if _recommender is None:
        from models.strategy_recommender.recommend import StrategyRecommender
        _recommender = StrategyRecommender()
    return _recommender


def classify_via_subprocess(text: str) -> dict:
    """Run Model 1 in a subprocess to avoid PyTorch/SQLite conflict."""
    code = f"""
import json
from models.event_classifier.predict import EventClassifier
clf = EventClassifier()
r = clf.predict({json.dumps(text)})
print(json.dumps(r))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=str(ROOT_DIR),
        timeout=30,
    )
    if result.returncode != 0:
        raise HTTPException(500, f"Classifier error: {result.stderr[:200]}")

    lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
    if not lines:
        raise HTTPException(500, "Classifier returned no output")
    return json.loads(lines[-1])


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "models": ["classifier", "exposure_scorer", "impact_estimator", "strategy_recommender"]}


@app.get("/stats")
def stats():
    from pipelines.utils import get_db_connection
    conn = get_db_connection()
    tables = {}
    for table in ["geopolitical_events", "company_financials", "financial_deltas",
                   "geopolitical_mentions", "corporate_impacts", "event_studies", "strategies"]:
        try:
            row = conn.execute(f"SELECT COUNT(*) as cnt FROM {table}").fetchone()
            tables[table] = row["cnt"]
        except Exception:
            tables[table] = 0
    conn.close()

    import csv
    with open(ROOT_DIR / "data" / "seed_labels" / "seed_labels.csv") as f:
        seed_count = sum(1 for _ in csv.DictReader(f))

    return {"database_tables": tables, "seed_labels": seed_count}


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    """Classify text into one of 8 geopolitical event categories."""
    return classify_via_subprocess(req.text)


@app.post("/exposure", response_model=ExposureResponse)
def exposure(req: ExposureRequest):
    """Score company exposure to a geopolitical event."""
    scorer = get_scorer()
    result = scorer.score(
        event_category=req.event_category,
        ticker=req.ticker,
        mention_sentiment=req.mention_sentiment,
        car_1_5=req.car_1_5,
    )
    return result


@app.post("/impact", response_model=ImpactResponse)
def impact(req: ImpactRequest):
    """Estimate financial impact range."""
    estimator = get_estimator()
    result = estimator.estimate(
        event_category=req.event_category,
        impact_channel=req.impact_channel,
        ticker=req.ticker,
        mention_sentiment=req.mention_sentiment,
        car_1_5=req.car_1_5,
        revenue_usd=req.revenue_usd,
    )
    return result


@app.post("/strategies")
def strategies(req: StrategyRequest):
    """Recommend ranked strategies for an event-channel combination."""
    recommender = get_recommender()
    results = recommender.recommend(
        event_category=req.event_category,
        impact_channel=req.impact_channel,
        severity=req.severity,
        company_size=req.company_size,
        top_n=req.top_n,
    )
    return {"strategies": results}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Full pipeline: classify → score exposure → estimate impact → recommend strategies."""

    # Step 1: Classify (subprocess)
    evt = classify_via_subprocess(req.text)

    # Step 2: Exposure
    scorer = get_scorer()
    exp = scorer.score(
        event_category=evt["category"],
        ticker=req.ticker,
        mention_sentiment=-0.5 if evt["confidence"] > 0.7 else -0.3,
    )

    # Step 3: Impact
    estimator = get_estimator()
    imp = estimator.estimate(
        event_category=evt["category"],
        impact_channel=exp["channel_prediction"],
        ticker=req.ticker,
        mention_sentiment=-0.5,
        revenue_usd=req.revenue_usd,
    )

    # Step 4: Strategies
    recommender = get_recommender()
    strats = recommender.recommend_full(
        event_category=evt["category"],
        top_channels=exp["top_3_channels"],
        severity=imp["impact_mid_pct"] / 100,
        company_size=req.company_size,
    )

    return {
        "event_classification": {
            "category": evt["category"],
            "confidence": evt["confidence"],
        },
        "exposure": {
            "primary_channel": exp["channel_prediction"],
            "channel_confidence": exp["channel_confidence"],
            "severity_score": exp["severity_score"],
            "top_3_channels": exp["top_3_channels"],
        },
        "impact_estimate": imp,
        "strategies": strats,
    }
