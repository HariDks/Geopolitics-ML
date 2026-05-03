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
import numpy as np

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
    """Geopolitical event text classifier.

    Uses ONNX Runtime with the pure Rust tokenizer by default — zero PyTorch
    dependency, no segfaults, runs in any process alongside SQLite.
    Falls back to PyTorch if ONNX model not found.
    """

    def __init__(self, model_path: str | Path | None = None, max_length: int = 256):
        path = Path(model_path) if model_path else MODEL_DIR
        self._max_length = max_length
        self._onnx_session = None
        self._torch_model = None
        self._device = None
        self._hf_tokenizer = None
        self._rust_tokenizer = None
        self._temperature = 1.0  # default: no scaling

        # Load temperature scaling calibration if available
        cal_path = path / "calibration.json"
        if cal_path.exists():
            import json as _json
            with open(cal_path) as f:
                self._temperature = _json.load(f).get("temperature", 1.0)

        onnx_path = path / "model.onnx"
        tokenizer_json = path / "tokenizer.json"

        if onnx_path.exists() and tokenizer_json.exists():
            try:
                # Pure ONNX path — no torch import at all
                import onnxruntime as ort
                from tokenizers import Tokenizer
                self._onnx_session = ort.InferenceSession(str(onnx_path))
                self._rust_tokenizer = Tokenizer.from_file(str(tokenizer_json))
                self._rust_tokenizer.enable_padding(length=max_length, pad_id=0)
                self._rust_tokenizer.enable_truncation(max_length=max_length)
                self._backend = "onnx"
            except Exception:
                # ONNX failed (version incompatibility, etc.) — try PyTorch
                self._try_pytorch(path)
        else:
            self._try_pytorch(path)

    def _try_pytorch(self, path):
        """Fallback to PyTorch if ONNX fails."""
        try:
            import torch
            from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
            self._hf_tokenizer = DistilBertTokenizer.from_pretrained(path)
            self._torch_model = DistilBertForSequenceClassification.from_pretrained(path)
            self._device = torch.device("cpu")
            self._torch_model.to(self._device)
            self._torch_model.eval()
            self._backend = "pytorch"
        except ImportError:
            # Neither ONNX nor PyTorch available — use keyword-based fallback
            self._backend = "keyword"

    _KEYWORD_MAP = {
        "trade_policy_actions": ["tariff", "import duty", "trade war", "customs", "anti-dumping", "quota", "trade agreement", "section 301", "export ban", "import ban"],
        "sanctions_financial_restrictions": ["sanction", "ofac", "asset freeze", "swift", "embargo", "blacklist", "sdn", "blocked persons"],
        "armed_conflict_instability": ["invasion", "war", "attack", "military", "bombing", "missile", "conflict", "terrorism", "killed", "troops"],
        "regulatory_sovereignty_shifts": ["regulation", "data localization", "cfius", "nationalization", "fdi", "local content", "privacy law", "compliance"],
        "technology_controls": ["export control", "chip", "semiconductor", "entity list", "bis", "dual-use", "huawei", "technology restriction"],
        "resource_energy_disruptions": ["opec", "oil price", "energy crisis", "commodity", "natural gas", "pipeline", "mining", "critical mineral", "lithium"],
        "political_transitions_volatility": ["election", "coup", "protest", "regime change", "president", "parliament", "constitutional", "populist"],
        "institutional_alliance_realignment": ["nato", "brexit", "eu withdrawal", "trade bloc", "brics", "alliance", "treaty", "wto"],
    }

    def _keyword_classify(self, text: str) -> np.ndarray:
        """Simple keyword-based classifier when neither ONNX nor PyTorch is available."""
        text_lower = text.lower()
        scores = np.zeros(len(CATEGORIES))
        for i, cat in enumerate(CATEGORIES):
            keywords = self._KEYWORD_MAP.get(cat, [])
            scores[i] = sum(1 for kw in keywords if kw in text_lower)
        if scores.sum() == 0:
            scores = np.ones(len(CATEGORIES))
        return self._softmax(scores * 3)  # amplify differences

    def _softmax(self, logits):
        """Compute temperature-scaled softmax from raw logits."""
        scaled = logits / self._temperature
        exp = np.exp(scaled - np.max(scaled))
        return exp / exp.sum()

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
        if self._backend == "keyword":
            probs = self._keyword_classify(text)
        elif self._backend == "onnx":
            enc = self._rust_tokenizer.encode(text)
            input_ids = np.array([enc.ids], dtype=np.int64)
            attention_mask = np.array([enc.attention_mask], dtype=np.int64)
            logits = self._onnx_session.run(None, {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            })[0]
            probs = self._softmax(logits.squeeze())
        else:
            import torch
            encoding = self._hf_tokenizer(
                text, truncation=True, padding="max_length",
                max_length=max_length, return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(self._device)
            attention_mask = encoding["attention_mask"].to(self._device)
            with torch.no_grad():
                outputs = self._torch_model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()

        pred_idx = probs.argmax()
        confidence = float(probs[pred_idx])
        all_scores = {cat: float(round(probs[i], 4)) for i, cat in enumerate(CATEGORIES)}

        # Confidence calibration: flag low-confidence predictions
        if confidence < 0.4:
            confidence_level = "low"
        elif confidence < 0.7:
            confidence_level = "moderate"
        else:
            confidence_level = "high"

        # Embedding backup: when primary classifier is uncertain, consult similar events
        if confidence_level == "low":
            try:
                from models.event_classifier.embedding_backup import EmbeddingBackup
                backup = EmbeddingBackup()
                backup_result = backup.classify(text, k=5)
                if backup_result["confidence"] >= 0.6:
                    # Embedding backup is more confident — use its answer
                    return {
                        "category": backup_result["category"],
                        "confidence": round(backup_result["confidence"], 4),
                        "confidence_level": "moderate",
                        "all_scores": all_scores,
                        "source": "embedding_backup",
                    }
            except Exception:
                pass  # Backup not available

        return {
            "category": CATEGORIES[pred_idx],
            "confidence": round(confidence, 4),
            "confidence_level": confidence_level,
            "all_scores": all_scores,
        }

    def predict_batch(self, texts: list[str], batch_size: int = 32, max_length: int = 256) -> list[dict]:
        """Classify a batch of texts efficiently."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            if self._backend == "keyword":
                probs = np.array([self._keyword_classify(t) for t in batch_texts])
            elif self._backend == "onnx":
                batch_probs = []
                for text in batch_texts:
                    enc = self._rust_tokenizer.encode(text)
                    input_ids = np.array([enc.ids], dtype=np.int64)
                    attention_mask = np.array([enc.attention_mask], dtype=np.int64)
                    logits = self._onnx_session.run(None, {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                    })[0]
                    batch_probs.append(self._softmax(logits.squeeze()))
                probs = np.array(batch_probs)
            else:
                import torch
                encodings = self._hf_tokenizer(
                    batch_texts, truncation=True, padding="max_length",
                    max_length=max_length, return_tensors="pt",
                )
                input_ids = encodings["input_ids"].to(self._device)
                attention_mask = encodings["attention_mask"].to(self._device)
                with torch.no_grad():
                    outputs = self._torch_model(input_ids=input_ids, attention_mask=attention_mask)
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
