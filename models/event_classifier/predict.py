"""
Model 1 inference: classify raw text into geopolitical event categories.

Usage:
    # As a module
    from models.event_classifier.predict import EventClassifier
    clf = EventClassifier()
    result = clf.predict("Russia invaded Ukraine, triggering NATO sanctions")
    # → {"category": "armed_conflict_instability", "confidence": 0.97, "all_scores": {...}}

    # From CLI
    python models/event_classifier/predict.py "US imposed 25% tariffs on Chinese steel imports"
    python models/event_classifier/predict.py --batch texts.txt
"""

import json
import sys
from pathlib import Path

import click
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

MODEL_DIR = Path(__file__).parent / "saved"

CATEGORIES = [
    "trade_policy_actions",
    "sanctions_financial_restrictions",
    "armed_conflict_instability",
    "regulatory_sovereignty_shifts",
    "technology_controls",
    "resource_energy_disruptions",
    "political_transitions_volatility",
    "institutional_alliance_realignment",
]


class EventClassifier:
    """Geopolitical event text classifier using fine-tuned DistilBERT."""

    def __init__(self, model_path: str | Path | None = None):
        path = Path(model_path) if model_path else MODEL_DIR
        self.tokenizer = DistilBertTokenizer.from_pretrained(path)
        self.model = DistilBertForSequenceClassification.from_pretrained(path)

        # Use CPU for pipeline compatibility — MPS can segfault when combined
        # with other models in the same process. MPS is used during training.
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str, max_length: int = 256) -> dict:
        """
        Classify a single text.

        Returns:
            {
                "category": "armed_conflict_instability",
                "confidence": 0.97,
                "all_scores": {"armed_conflict_instability": 0.97, ...}
            }
        """
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()

        pred_idx = probs.argmax()
        all_scores = {cat: float(round(probs[i], 4)) for i, cat in enumerate(CATEGORIES)}

        return {
            "category": CATEGORIES[pred_idx],
            "confidence": float(round(probs[pred_idx], 4)),
            "all_scores": all_scores,
        }

    def predict_batch(self, texts: list[str], batch_size: int = 32, max_length: int = 256) -> list[dict]:
        """Classify a batch of texts efficiently."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            for j, p in enumerate(probs):
                pred_idx = p.argmax()
                all_scores = {cat: float(round(p[k], 4)) for k, cat in enumerate(CATEGORIES)}
                results.append({
                    "text": batch_texts[j][:100],
                    "category": CATEGORIES[pred_idx],
                    "confidence": float(round(p[pred_idx], 4)),
                    "all_scores": all_scores,
                })

        return results


@click.command()
@click.argument("text", required=False)
@click.option("--batch", type=click.Path(exists=True), help="File with one text per line")
@click.option("--model-path", default=None, help="Path to saved model")
def main(text, batch, model_path):
    """Classify geopolitical event text."""
    clf = EventClassifier(model_path)

    if batch:
        with open(batch) as f:
            texts = [line.strip() for line in f if line.strip()]
        results = clf.predict_batch(texts)
        for r in results:
            print(f"{r['category']:40s} ({r['confidence']:.3f})  {r['text']}")
    elif text:
        result = clf.predict(text)
        print(json.dumps(result, indent=2))
    else:
        # Interactive mode: demo with sample texts
        samples = [
            "US imposed 25% tariffs on Chinese steel and aluminum imports effective immediately",
            "Russian forces launched a full-scale invasion of Ukraine from multiple directions",
            "EU passed the Digital Markets Act requiring Big Tech companies to open their platforms",
            "OFAC added 15 Russian oligarchs to the Specially Designated Nationals list",
            "BIS restricted exports of advanced AI chips to China including NVIDIA A100 and H100",
            "OPEC announced a surprise production cut of 1.16 million barrels per day",
            "Military coup in Myanmar: army detained Aung San Suu Kyi and seized power",
            "UK formally withdrew from the European Union triggering Article 50",
            "Houthi rebels attacked container ship in Red Sea forcing Maersk to reroute via Cape of Good Hope",
            "India banned 59 Chinese apps including TikTok and WeChat citing national security",
        ]
        print("Model 1 — Event Classifier Demo\n" + "=" * 60)
        results = clf.predict_batch(samples)
        for r in results:
            print(f"  {r['category']:40s} ({r['confidence']:.3f})  {r['text']}")


if __name__ == "__main__":
    main()
