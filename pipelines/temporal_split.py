"""
Temporal train/val/test splitting for all models.

Splits data by event date to prevent future information leakage:
  - Train: events from 2015-2022
  - Validation: events from 2023
  - Test: events from 2024-2026

This is the only honest way to measure how well models handle new events.
Random splits leak future information and inflate accuracy.

Usage:
    from pipelines.temporal_split import get_temporal_split, extract_year

    split = get_temporal_split(event_id, quarter)
    # Returns: "train", "val", or "test"
"""

import re


def extract_year(event_id: str, quarter: str = "") -> int | None:
    """Extract the year from a seed label's event_id or quarter field."""
    # Try quarter first (e.g., '2022Q1', '2023FY')
    if quarter and len(quarter) >= 4 and quarter[:4].isdigit():
        return int(quarter[:4])

    # Try event_id (e.g., 'russia_invasion_2022', 'covid_lockdown_start')
    for year in range(2008, 2027):
        if str(year) in event_id:
            return year

    # Fallback: map known events without year in name
    known_events = {
        "covid_lockdown_start": 2020,
        "suez_blockage_2021": 2021,
    }
    for key, year in known_events.items():
        if key in event_id:
            return year

    return None


def get_temporal_split(event_id: str, quarter: str = "") -> str:
    """
    Assign a label to train/val/test based on event date.

    Returns:
        "train" for events up to 2022
        "val" for events in 2023
        "test" for events in 2024+
        "train" as default if year can't be determined
    """
    year = extract_year(event_id, quarter)
    if year is None:
        return "train"  # conservative default
    if year <= 2022:
        return "train"
    elif year == 2023:
        return "val"
    else:
        return "test"


def split_seed_labels(labels: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Split seed labels into train/val/test by event date."""
    train, val, test = [], [], []
    for label in labels:
        split = get_temporal_split(label["event_id"], label.get("quarter", ""))
        if split == "train":
            train.append(label)
        elif split == "val":
            val.append(label)
        else:
            test.append(label)
    return train, val, test


def split_event_studies(event_studies: list[dict], event_dates: dict[str, str]) -> tuple[list, list, list]:
    """Split event studies by event date."""
    train, val, test = [], [], []
    for es in event_studies:
        date = event_dates.get(es.get("event_id", ""), "")
        year = int(date[:4]) if date and len(date) >= 4 else None
        if year is None or year <= 2022:
            train.append(es)
        elif year == 2023:
            val.append(es)
        else:
            test.append(es)
    return train, val, test


if __name__ == "__main__":
    import csv
    from pathlib import Path

    SEED_PATH = Path(__file__).parent.parent / "data" / "seed_labels" / "seed_labels.csv"
    with open(SEED_PATH) as f:
        labels = list(csv.DictReader(f))

    train, val, test = split_seed_labels(labels)

    print(f"Temporal split results:")
    print(f"  Train (<=2022): {len(train)} labels")
    print(f"  Val (2023):     {len(val)} labels")
    print(f"  Test (2024+):   {len(test)} labels")
    print(f"  Total:          {len(labels)}")

    # Show manual vs auto in each split
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        manual = sum(1 for r in split if r.get("labeled_by", "") in ("claude_verified", "human_review", ""))
        auto = len(split) - manual
        print(f"  {name}: {manual} manual + {auto} auto = {len(split)}")
