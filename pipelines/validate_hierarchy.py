"""
Validate and map seed labels against the event hierarchy.

This script:
1. Maps every seed label event_id to a (crisis_id, episode_id) pair
2. Identifies orphan event_ids that don't belong to any crisis
3. Detects duplicate event_ids that should be merged
4. Reports event granularity stats

Usage:
    python pipelines/validate_hierarchy.py
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

HIERARCHY_PATH = ROOT_DIR / "data" / "mappings" / "event_hierarchy.json"
SEED_PATH = ROOT_DIR / "data" / "seed_labels" / "seed_labels.csv"


def load_hierarchy():
    with open(HIERARCHY_PATH) as f:
        data = json.load(f)
    return data["crises"]


def build_event_id_map(crises):
    """Build event_id → (crisis_id, episode_id) mapping."""
    mapping = {}
    for crisis in crises:
        for episode in crisis["episodes"]:
            for event_id in episode["seed_label_event_ids"]:
                mapping[event_id] = {
                    "crisis_id": crisis["crisis_id"],
                    "crisis_name": crisis["crisis_name"],
                    "episode_id": episode["episode_id"],
                    "episode_type": episode["episode_type"],
                    "canonical_date": episode["episode_date"],
                }
    return mapping


def validate():
    crises = load_hierarchy()
    event_map = build_event_id_map(crises)

    with open(SEED_PATH) as f:
        labels = list(csv.DictReader(f))

    # Map every label
    mapped = 0
    orphans = Counter()
    crisis_counts = Counter()
    episode_counts = Counter()
    duplicate_ids = Counter()

    all_event_ids = Counter(r["event_id"] for r in labels)

    for label in labels:
        eid = label["event_id"]
        if eid in event_map:
            mapped += 1
            crisis_counts[event_map[eid]["crisis_name"]] += 1
            episode_counts[event_map[eid]["episode_id"]] += 1
        else:
            orphans[eid] += 1

    # Find event_ids that should be merged (same episode, different names)
    episode_to_ids = {}
    for eid, info in event_map.items():
        ep = info["episode_id"]
        if ep not in episode_to_ids:
            episode_to_ids[ep] = []
        episode_to_ids[ep].append(eid)

    duplicates = {ep: ids for ep, ids in episode_to_ids.items() if len(ids) > 1}

    # Report
    print("=" * 80)
    print("EVENT HIERARCHY VALIDATION")
    print("=" * 80)
    print(f"\nSeed labels: {len(labels)}")
    print(f"Mapped to hierarchy: {mapped} ({mapped/len(labels):.0%})")
    print(f"Orphans (no crisis assigned): {len(orphans)} unique event_ids, {sum(orphans.values())} labels")

    print(f"\nCRISES IN HIERARCHY: {len(crises)}")
    for crisis in crises:
        n_episodes = len(crisis["episodes"])
        n_labels = crisis_counts[crisis["crisis_name"]]
        print(f"  {crisis['crisis_id']:40s} {n_episodes} episodes, {n_labels} labels")

    if duplicates:
        print(f"\nDUPLICATE EVENT_IDS (same episode, different names — should be merged):")
        for ep, ids in duplicates.items():
            print(f"  {ep}:")
            for eid in ids:
                count = all_event_ids.get(eid, 0)
                print(f"    {eid} ({count} labels)")

    if orphans:
        print(f"\nORPHAN EVENT_IDS ({len(orphans)} — not in any crisis):")
        for eid, count in orphans.most_common(20):
            print(f"  {eid:50s} {count} labels")
        if len(orphans) > 20:
            print(f"  ... and {len(orphans) - 20} more")

    # Granularity analysis
    print(f"\n{'=' * 80}")
    print(f"GRANULARITY ANALYSIS")
    print(f"{'=' * 80}")
    print(f"\nTotal crises defined: {len(crises)}")
    print(f"Total episodes defined: {sum(len(c['episodes']) for c in crises)}")
    print(f"Total observations (GDELT+ACLED rows for mapped crises): ~{mapped * 50}+")
    print(f"\nAvg labels per crisis: {mapped / max(len(crisis_counts), 1):.1f}")
    print(f"Avg labels per episode: {mapped / max(len(episode_counts), 1):.1f}")

    print(f"\nRECOMMENDATIONS:")
    if orphans:
        print(f"  1. Create crises for the {len(orphans)} orphan event_ids")
    if duplicates:
        print(f"  2. Merge {len(duplicates)} sets of duplicate event_ids into canonical names")
    print(f"  3. Use crisis_id as the primary grouping key for backtest and analysis")
    print(f"  4. Use episode_id for fine-grained impact timing")


if __name__ == "__main__":
    validate()
