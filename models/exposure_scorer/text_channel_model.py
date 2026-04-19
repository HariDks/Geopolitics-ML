"""
Text-based channel prediction model.

Uses TF-IDF + Logistic Regression to predict impact channel from mention_text.
This captures the linguistic signal that XGBoost misses — words like
"impairment" → capital_allocation, "supply disruption" → procurement.

Can be used standalone or combined with the XGBoost structured model
for a hybrid prediction.

Usage:
    python models/exposure_scorer/text_channel_model.py           # train + eval
    python models/exposure_scorer/text_channel_model.py --eval-only
"""

import csv
import json
import pickle
import sys
from pathlib import Path

import click
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

MODEL_DIR = Path(__file__).parent / "saved"
SEED_PATH = ROOT_DIR / "data" / "seed_labels" / "seed_labels.csv"

IMPACT_CHANNELS = [
    "procurement_supply_chain", "revenue_market_access",
    "capital_allocation_investment", "regulatory_compliance_cost",
    "logistics_operations", "innovation_ip", "workforce_talent",
    "reputation_stakeholder", "financial_treasury", "cybersecurity_it",
]
CH2IDX = {c: i for i, c in enumerate(IMPACT_CHANNELS)}


def load_data():
    """Load seed labels with text."""
    with open(SEED_PATH) as f:
        labels = list(csv.DictReader(f))

    texts = []
    channels = []
    is_manual = []

    for r in labels:
        text = r.get("mention_text", "").strip()
        channel = r.get("impact_channel", "")
        if len(text) < 20 or channel not in CH2IDX:
            continue
        texts.append(text)
        channels.append(CH2IDX[channel])
        is_manual.append(r.get("labeled_by", "") in ("claude_verified", "human_review", ""))

    return texts, channels, is_manual


def train_text_model(texts, channels, is_manual):
    """Train TF-IDF + Logistic Regression for channel prediction."""

    # Split: use manual labels as holdout, train on all data
    manual_idx = [i for i, m in enumerate(is_manual) if m]
    auto_idx = [i for i, m in enumerate(is_manual) if not m]

    # ONLY train on manual labels to avoid data leakage.
    # Auto-generated labels have mention_text containing channel names
    # (e.g., "Sector-targeted auto-label: BA (workforce_talent)...")
    # which lets the model cheat by reading the answer from the text.
    manual_texts = [texts[i] for i in manual_idx]
    manual_channels = [channels[i] for i in manual_idx]
    X_train_texts, X_val_texts, y_train, y_val = train_test_split(
        manual_texts, manual_channels, test_size=0.25, stratify=manual_channels, random_state=42,
    )

    # TF-IDF: capture channel-indicative phrases
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),  # unigrams, bigrams, trigrams
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(X_train_texts)
    X_val = vectorizer.transform(X_val_texts)

    # Logistic Regression with class weights
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Dev eval
    y_pred = model.predict(X_val)
    print("=" * 70)
    print("TEXT CHANNEL MODEL — Dev Evaluation (random 20% split)")
    print("=" * 70)
    print(classification_report(
        y_val, y_pred,
        target_names=IMPACT_CHANNELS,
        digits=3,
        zero_division=0,
    ))

    # Holdout eval (manual labels only)
    manual_texts = [texts[i] for i in manual_idx]
    manual_channels = [channels[i] for i in manual_idx]

    if manual_texts:
        X_manual = vectorizer.transform(manual_texts)
        y_manual_pred = model.predict(X_manual)
        y_manual_proba = model.predict_proba(X_manual)

        # Top-1, top-2, top-3
        top1 = sum(1 for p, a in zip(y_manual_pred, manual_channels) if p == a)
        top2 = sum(1 for proba, actual in zip(y_manual_proba, manual_channels)
                   if actual in proba.argsort()[-2:])
        top3 = sum(1 for proba, actual in zip(y_manual_proba, manual_channels)
                   if actual in proba.argsort()[-3:])
        n = len(manual_channels)

        print("=" * 70)
        print("TEXT CHANNEL MODEL — Holdout Evaluation (163 manual labels)")
        print("=" * 70)
        print(f"  Top-1 accuracy:  {top1}/{n} ({top1/n:.1%})")
        print(f"  Top-2 accuracy:  {top2}/{n} ({top2/n:.1%})")
        print(f"  Top-3 accuracy:  {top3}/{n} ({top3/n:.1%})")

        print(f"\n  Comparison to XGBoost structured model:")
        print(f"  {'Metric':20s} {'XGBoost':>10s} {'Text Model':>12s} {'Delta':>8s}")
        print(f"  {'-'*55}")
        xgb_top1, xgb_top2, xgb_top3 = 0.507, 0.609, 0.739
        print(f"  {'Top-1':20s} {xgb_top1:>9.1%} {top1/n:>11.1%} {(top1/n - xgb_top1):>+7.1%}")
        print(f"  {'Top-2':20s} {xgb_top2:>9.1%} {top2/n:>11.1%} {(top2/n - xgb_top2):>+7.1%}")
        print(f"  {'Top-3':20s} {xgb_top3:>9.1%} {top3/n:>11.1%} {(top3/n - xgb_top3):>+7.1%}")

    # Show top features per channel
    print(f"\nTOP PREDICTIVE PHRASES PER CHANNEL:")
    feature_names = vectorizer.get_feature_names_out()
    for i, channel in enumerate(IMPACT_CHANNELS):
        if i < model.coef_.shape[0]:
            top_idx = model.coef_[i].argsort()[-5:][::-1]
            top_words = [feature_names[j] for j in top_idx]
            print(f"  {channel[:30]:30s} {', '.join(top_words)}")

    return model, vectorizer


@click.command()
@click.option("--eval-only", is_flag=True)
def main(eval_only):
    texts, channels, is_manual = load_data()
    print(f"Loaded {len(texts)} labels with text")

    model, vectorizer = train_text_model(texts, channels, is_manual)

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "text_channel_model.pkl", "wb") as f:
        pickle.dump({"model": model, "vectorizer": vectorizer}, f)
    print(f"\nSaved to {MODEL_DIR / 'text_channel_model.pkl'}")


if __name__ == "__main__":
    main()
