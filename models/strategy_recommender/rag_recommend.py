"""
RAG-based strategy recommender — retrieves similar historical cases
and synthesizes grounded recommendations with citations.

Instead of generic strategies ("route diversification planning"),
this produces recommendations like:
  "Maersk rerouted via Cape of Good Hope in 2024, adding 14 days
   but benefiting from rate surge. Consider similar rerouting with
   pre-negotiated alternative port contracts."

Uses sentence-transformers for embedding + ChromaDB for retrieval.
LLM synthesis is optional (falls back to template-based if no API key).

Usage:
    from models.strategy_recommender.rag_recommend import RAGRecommender

    rec = RAGRecommender()
    rec.build_index()  # one-time
    results = rec.recommend(
        event_text="Houthi rebels attacked Red Sea shipping",
        company="Costco",
        sector="Consumer Staples",
        channel="procurement_supply_chain",
    )
"""

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pipelines.utils import get_logger

logger = get_logger("rag_recommend")

MODEL_DIR = Path(__file__).parent
CASE_STUDIES_PATH = MODEL_DIR / "case_studies.json"
CHROMA_DIR = MODEL_DIR / "chroma_cases"


class RAGRecommender:
    """Retrieval-Augmented strategy recommender with case study citations."""

    def __init__(self):
        self._model = None
        self._collection = None
        self._cases = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def _load_cases(self):
        if self._cases is None:
            with open(CASE_STUDIES_PATH) as f:
                data = json.load(f)
            self._cases = data["cases"]

    def _get_collection(self):
        if self._collection is None:
            import chromadb
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            self._collection = client.get_or_create_collection(
                name="case_studies",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def build_index(self, force: bool = False):
        """Embed all case studies and store in ChromaDB."""
        self._load_model()
        self._load_cases()
        collection = self._get_collection()

        if collection.count() > 0 and not force:
            logger.info(f"Index already has {collection.count()} cases. Use force=True to rebuild.")
            return

        # Create searchable text for each case
        texts = []
        ids = []
        metadatas = []
        for i, case in enumerate(self._cases):
            # Combine key fields into searchable document
            doc = (
                f"{case['event']}. {case['company']} ({case['year']}). "
                f"Category: {case['category']}. Channel: {case['channel']}. "
                f"Action: {case['action']} Outcome: {case['outcome']}"
            )
            texts.append(doc)
            ids.append(f"case-{i}")
            metadatas.append({
                "company": case["company"],
                "year": str(case["year"]),
                "category": case["category"],
                "channel": case["channel"],
            })

        logger.info(f"Embedding {len(texts)} case studies...")
        embeddings = self._model.encode(texts, show_progress_bar=False)

        collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(f"Indexed {collection.count()} case studies")

    def recommend(
        self,
        event_text: str,
        company: str = "",
        sector: str = "",
        channel: str = "",
        k: int = 5,
    ) -> list[dict]:
        """
        Retrieve similar cases and generate grounded recommendations.

        Returns list of recommendations, each with:
        - strategy: what to do
        - precedent: which historical case supports it
        - company_example: who did it
        - outcome: what happened
        - relevance: how similar the precedent is
        """
        self._load_model()
        self._load_cases()
        collection = self._get_collection()

        if collection.count() == 0:
            self.build_index()

        # Build query combining event + company context
        query = f"{event_text}. Company: {company}. Sector: {sector}. Channel: {channel}."
        query_embedding = self._model.encode([query])[0]

        # Retrieve top-k similar cases
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(k * 2, collection.count()),  # retrieve extra, then filter
        )

        recommendations = []
        seen_strategies = set()

        for i in range(len(results["ids"][0])):
            case_idx = int(results["ids"][0][i].split("-")[1])
            case = self._cases[case_idx]
            distance = results["distances"][0][i] if results.get("distances") else 0

            # Extract the core strategy from the action
            action = case["action"]
            strategy_key = action[:50]  # deduplicate by first 50 chars
            if strategy_key in seen_strategies:
                continue
            seen_strategies.add(strategy_key)

            # Generate recommendation from precedent
            rec = {
                "strategy": _extract_strategy(case, channel),
                "precedent": {
                    "company": case["company"],
                    "year": case["year"],
                    "event": case["event"],
                    "action": case["action"],
                    "outcome": case["outcome"],
                    "cost": case.get("cost", "Unknown"),
                    "timeline": case.get("timeline", "Unknown"),
                },
                "relevance": round(1 - distance, 2) if distance else 0.5,
            }
            recommendations.append(rec)

            if len(recommendations) >= k:
                break

        return recommendations


def _extract_strategy(case: dict, target_channel: str) -> str:
    """Generate a strategy recommendation from a historical case."""
    company = case["company"]
    action = case["action"]
    outcome = case["outcome"]
    channel = case["channel"]

    # Template-based synthesis (no LLM needed)
    if "exit" in action.lower() or "sold" in action.lower() or "divest" in action.lower():
        return f"Consider structured exit or divestiture. {company} ({case['year']}) executed a controlled exit, taking a one-time charge rather than ongoing exposure."

    if "reroute" in action.lower() or "route" in action.lower():
        return f"Evaluate alternative routing or supply paths. {company} ({case['year']}) rerouted operations, which increased costs but maintained business continuity."

    if "diversif" in action.lower() or "india" in action.lower() or "vietnam" in action.lower():
        return f"Accelerate geographic diversification. {company} ({case['year']}) shifted operations to alternative locations, reducing concentration risk."

    if "relocat" in action.lower() or "evacuat" in action.lower():
        return f"Prepare workforce relocation and remote work protocols. {company} ({case['year']}) relocated employees while maintaining client relationships."

    if "ransom" in action.lower() or "cyber" in action.lower() or "rebuild" in action.lower():
        return f"Strengthen cybersecurity defenses and incident response. {company} ({case['year']}) rebuilt systems after an attack. Consider cyber insurance with explicit coverage for geopolitical attacks."

    if "inventory" in action.lower() or "stockpil" in action.lower() or "pre-position" in action.lower():
        return f"Pre-position critical inventory. {company} ({case['year']}) built safety stock ahead of disruption, avoiding supply gaps."

    if "compliance" in action.lower() or "workaround" in action.lower() or "develop" in action.lower():
        return f"Develop compliant alternatives. {company} ({case['year']}) created product variants that met regulatory requirements while preserving market access."

    if "team" in action.lower() or "task force" in action.lower() or "framework" in action.lower():
        return f"Establish a dedicated geopolitical risk function. {company} ({case['year']}) built an internal team that identified and mitigated risks proactively."

    # Generic fallback
    return f"Learn from {company}'s response ({case['year']}): {action[:150]}"


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--build", is_flag=True, help="Build case study index")
    @click.option("--test", default=None, help="Test with event text")
    def main(build, test):
        rec = RAGRecommender()

        if build:
            rec.build_index(force=True)

        if test:
            results = rec.recommend(event_text=test, k=5)
            print(f"\nRecommendations for: {test[:60]}...\n")
            for i, r in enumerate(results):
                print(f"#{i+1} {r['strategy']}")
                print(f"   Precedent: {r['precedent']['company']} ({r['precedent']['year']}) — {r['precedent']['event']}")
                print(f"   Outcome: {r['precedent']['outcome'][:100]}")
                print(f"   Relevance: {r['relevance']}")
                print()

    main()
