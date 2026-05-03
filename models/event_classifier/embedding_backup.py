"""
Embedding-based backup classifier for the event classifier.

When DistilBERT/ONNX confidence is low (< threshold), this module
finds the 5 most similar past events by embedding similarity and
uses their categories as a vote.

Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings and
ChromaDB for vector storage/retrieval.

The key insight: when a model says "I'm not sure if this is
armed_conflict or political_transitions," finding 5 similar
historical events and checking what THEY were classified as
often gives a better answer than the uncertain model.

Usage:
    from models.event_classifier.embedding_backup import EmbeddingBackup

    backup = EmbeddingBackup()
    backup.build_index()  # one-time: embed all labeled events
    result = backup.classify("Houthi rebels fired missiles at shipping", k=5)
    # → {"category": "armed_conflict_instability", "confidence": 0.8, "votes": {...}}
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pipelines.utils import get_logger

logger = get_logger("embedding_backup")

MODEL_DIR = Path(__file__).parent / "saved"
CHROMA_DIR = MODEL_DIR / "chroma_events"

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


class EmbeddingBackup:
    """Embedding-similarity classifier using sentence-transformers + ChromaDB."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = None
        self._collection = None
        self._model_name = model_name

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)

    def _get_collection(self):
        if self._collection is None:
            import chromadb
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            self._collection = client.get_or_create_collection(
                name="event_texts",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def build_index(self, force: bool = False):
        """
        Build the embedding index from all labeled events in the database.

        Sources:
        - ACLED descriptions (with categories)
        - GTA descriptions (with categories)
        - OFAC descriptions (sanctions)
        - BIS descriptions (technology_controls)
        - Seed label mention_text (with categories)
        - News augmentation examples (with categories)
        """
        collection = self._get_collection()

        # Check if already built
        if collection.count() > 0 and not force:
            logger.info(f"Index already has {collection.count()} entries. Use force=True to rebuild.")
            return

        self._load_model()

        texts = []
        categories = []
        ids = []

        # Source 1: Database events with descriptions
        try:
            from pipelines.utils import get_db_connection
            conn = get_db_connection()

            for source, cat_col in [("acled", "event_category"), ("gta", "event_category"),
                                      ("ofac", "event_category"), ("bis", "event_category")]:
                rows = conn.execute(f"""
                    SELECT event_id, description_text, {cat_col} as category
                    FROM geopolitical_events
                    WHERE source = ? AND description_text IS NOT NULL
                    AND LENGTH(description_text) > 30
                    AND event_date <= '2022-12-31'
                    ORDER BY RANDOM() LIMIT 500
                """, (source,)).fetchall()

                for r in rows:
                    texts.append(r["description_text"][:512])
                    categories.append(r["category"])
                    ids.append(f"db-{r['event_id']}")

            conn.close()
        except Exception as e:
            logger.warning(f"Database not available: {e}")

        # Source 2: News augmentation examples
        news_path = ROOT_DIR / "data" / "seed_labels" / "news_augmentation.json"
        if news_path.exists():
            with open(news_path) as f:
                news_data = json.load(f)
            for cat, examples in news_data.items():
                if cat in CATEGORIES:
                    for i, text in enumerate(examples):
                        texts.append(text)
                        categories.append(cat)
                        ids.append(f"news-{cat}-{i}")

        # Source 3: Seed labels with mention_text
        import csv
        seed_path = ROOT_DIR / "data" / "seed_labels" / "seed_labels.csv"
        if seed_path.exists():
            from models.exposure_scorer.train import EVENT_TO_CATEGORY
            with open(seed_path) as f:
                for i, r in enumerate(csv.DictReader(f)):
                    text = r.get("mention_text", "").strip()
                    event_cat = EVENT_TO_CATEGORY.get(r.get("event_id", ""), "")
                    if text and len(text) > 30 and event_cat:
                        texts.append(text[:512])
                        categories.append(event_cat)
                        ids.append(f"seed-{i}")

        if not texts:
            logger.error("No texts to index")
            return

        logger.info(f"Embedding {len(texts)} texts...")
        embeddings = self._model.encode(texts, show_progress_bar=True, batch_size=64)

        # Clear and rebuild
        if collection.count() > 0:
            collection.delete(where={"category": {"$ne": ""}})

        # Add in batches (ChromaDB has batch size limits)
        batch_size = 500
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            collection.add(
                embeddings=embeddings[start:end].tolist(),
                documents=texts[start:end],
                metadatas=[{"category": c} for c in categories[start:end]],
                ids=ids[start:end],
            )

        logger.info(f"Indexed {collection.count()} events in ChromaDB")

    def classify(self, text: str, k: int = 5) -> dict:
        """
        Classify text by finding k most similar past events and voting.

        Returns:
            {
                "category": "armed_conflict_instability",
                "confidence": 0.8,  # 4/5 agreed
                "votes": {"armed_conflict_instability": 4, "political_transitions_volatility": 1},
                "similar_events": [...]
            }
        """
        self._load_model()
        collection = self._get_collection()

        if collection.count() == 0:
            return {"category": CATEGORIES[0], "confidence": 0.0, "votes": {}, "similar_events": []}

        # Embed the query
        query_embedding = self._model.encode([text])[0]

        # Find k nearest neighbors
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
        )

        # Vote on category
        votes = Counter()
        similar = []
        for i in range(len(results["ids"][0])):
            cat = results["metadatas"][0][i]["category"]
            doc = results["documents"][0][i][:100]
            dist = results["distances"][0][i] if results.get("distances") else None
            votes[cat] += 1
            similar.append({"category": cat, "text": doc, "distance": dist})

        if not votes:
            return {"category": CATEGORIES[0], "confidence": 0.0, "votes": {}, "similar_events": []}

        top_cat = votes.most_common(1)[0][0]
        confidence = votes[top_cat] / k

        return {
            "category": top_cat,
            "confidence": round(confidence, 2),
            "votes": dict(votes),
            "similar_events": similar,
        }


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--build", is_flag=True, help="Build the embedding index")
    @click.option("--force", is_flag=True, help="Force rebuild even if index exists")
    @click.option("--test", default=None, help="Test classify a text")
    def main(build, force, test):
        backup = EmbeddingBackup()

        if build:
            backup.build_index(force=force)

        if test:
            result = backup.classify(test)
            print(f"Category: {result['category']} ({result['confidence']:.0%})")
            print(f"Votes: {result['votes']}")
            for s in result["similar_events"]:
                print(f"  [{s['category'][:20]}] {s['text']}")

    main()
