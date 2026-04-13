"""
Model 1: Event Classifier — Fine-tuned DistilBERT for geopolitical event taxonomy.

Classifies raw text into 8 event categories:
  0: trade_policy_actions
  1: sanctions_financial_restrictions
  2: armed_conflict_instability
  3: regulatory_sovereignty_shifts
  4: technology_controls
  5: resource_energy_disruptions
  6: political_transitions_volatility
  7: institutional_alliance_realignment

Training data: ACLED descriptions, GTA intervention titles, OFAC SDN entries,
BIS entity list entries, and non-boilerplate EDGAR filing mentions.

Usage:
    python models/event_classifier/train.py
    python models/event_classifier/train.py --epochs 5 --batch-size 32
    python models/event_classifier/train.py --eval-only --model-path models/event_classifier/saved
"""

import json
import os
import sys
from pathlib import Path

import click
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pipelines.utils import get_db_connection, get_logger

logger = get_logger("event_classifier")

ROOT_DIR = Path(__file__).parent.parent.parent
MODEL_DIR = Path(__file__).parent / "saved"

# Label mapping
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
CAT2IDX = {c: i for i, c in enumerate(CATEGORIES)}
IDX2CAT = {i: c for c, i in CAT2IDX.items()}

# Per-category sample limits for balanced training
MAX_PER_CATEGORY = 5000
MIN_PER_CATEGORY = 50  # below this, we augment


class EventTextDataset(Dataset):
    """Simple text + label dataset for the classifier."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_training_data(conn, max_per_cat: int = MAX_PER_CATEGORY) -> tuple[list[str], list[int]]:
    """
    Load balanced training data from DB sources.

    Strategy:
    - ACLED: rich text descriptions (armed_conflict, political_transitions, institutional)
    - GTA: intervention titles (trade_policy, regulatory, technology_controls)
    - OFAC: SDN addition descriptions (sanctions)
    - BIS: entity list entries (technology_controls)
    - EDGAR mentions: non-boilerplate filing text (all categories, specificity > 30)

    Downsample large categories, use all records for small ones.
    """
    texts = []
    labels = []
    category_counts = {c: 0 for c in CATEGORIES}

    # ── ACLED: best source for armed_conflict, political_transitions, institutional ──
    for cat in ["armed_conflict_instability", "political_transitions_volatility",
                "institutional_alliance_realignment"]:
        rows = conn.execute(
            """SELECT description_text FROM geopolitical_events
               WHERE source = 'acled'
               AND event_category = ?
               AND description_text IS NOT NULL
               AND LENGTH(description_text) > 30
               ORDER BY RANDOM()
               LIMIT ?""",
            (cat, max_per_cat),
        ).fetchall()

        for r in rows:
            texts.append(r["description_text"][:512])
            labels.append(CAT2IDX[cat])
            category_counts[cat] += 1

    # ── GTA: best source for trade_policy, regulatory, technology ──
    for cat in ["trade_policy_actions", "regulatory_sovereignty_shifts", "technology_controls"]:
        rows = conn.execute(
            """SELECT description_text FROM geopolitical_events
               WHERE source = 'gta'
               AND event_category = ?
               AND description_text IS NOT NULL
               AND LENGTH(description_text) > 10
               ORDER BY RANDOM()
               LIMIT ?""",
            (cat, max_per_cat),
        ).fetchall()

        for r in rows:
            texts.append(r["description_text"][:512])
            labels.append(CAT2IDX[cat])
            category_counts[cat] += 1

    # ── OFAC: sanctions ──
    rows = conn.execute(
        """SELECT description_text FROM geopolitical_events
           WHERE source = 'ofac'
           AND description_text IS NOT NULL
           AND LENGTH(description_text) > 20
           ORDER BY RANDOM()
           LIMIT ?""",
        (max_per_cat,),
    ).fetchall()
    for r in rows:
        texts.append(r["description_text"][:512])
        labels.append(CAT2IDX["sanctions_financial_restrictions"])
        category_counts["sanctions_financial_restrictions"] += 1

    # ── BIS: technology controls ──
    rows = conn.execute(
        """SELECT description_text FROM geopolitical_events
           WHERE source = 'bis'
           AND description_text IS NOT NULL
           AND LENGTH(description_text) > 10
           ORDER BY RANDOM()
           LIMIT ?""",
        (max_per_cat,),
    ).fetchall()
    for r in rows:
        texts.append(r["description_text"][:512])
        labels.append(CAT2IDX["technology_controls"])
        category_counts["technology_controls"] += 1

    # ── EDGAR mentions: non-boilerplate, augments all categories ──
    rows = conn.execute(
        """SELECT mention_text, primary_category FROM geopolitical_mentions
           WHERE specificity_score > 30
           AND primary_category IN ({})
           ORDER BY specificity_score DESC""".format(
            ",".join(f"'{c}'" for c in CATEGORIES)
        ),
    ).fetchall()
    for r in rows:
        cat = r["primary_category"]
        if cat in CAT2IDX and category_counts.get(cat, 0) < max_per_cat:
            texts.append(r["mention_text"][:512])
            labels.append(CAT2IDX[cat])
            category_counts[cat] += 1

    # ── Synthetic augmentation for resource_energy (0 text records in DB) ──
    # Use seed label mention_text from our CSV + hand-crafted examples
    resource_energy_examples = [
        "OPEC announced a surprise production cut of 1.16 million barrels per day, sending oil prices surging 6%.",
        "China restricted exports of gallium and germanium, critical minerals used in semiconductor manufacturing.",
        "European natural gas prices surged 400% as Russia reduced pipeline flows through Nord Stream.",
        "Indonesia banned nickel ore exports to force downstream processing, disrupting global battery supply chains.",
        "Chile announced nationalization of its lithium industry, affecting SQM and Albemarle operations.",
        "Ukraine grain crisis: Black Sea blockade halted 25 million tonnes of wheat and corn exports.",
        "Saudi Arabia and Russia launched an oil price war, crashing Brent crude from $60 to $20 per barrel.",
        "Critical mineral supply chain disruption as DRC suspended cobalt exports over royalty disputes.",
        "Australia-China trade war: Beijing imposed 80% tariffs on Australian barley and banned coal imports.",
        "Energy security concerns mount as Strait of Hormuz tensions threaten 21% of global oil transit.",
        "Global LNG shortage intensified as Ras Laffan facility in Qatar suffered major attack, losing 17% capacity.",
        "Rare earth export controls by China target US defense supply chain, affecting F-35 production.",
        "Oil supply disruption: Nigerian oil theft reduced production by 400,000 barrels per day.",
        "Libya civil war shut down the Sharara oil field, removing 300,000 bpd from global supply.",
        "South Africa's Eskom load-shedding crisis cost mining companies $80-110M in lost production.",
        "Global fertilizer prices tripled after Belarus sanctions and Russia export ban disrupted potash markets.",
        "Pakistan energy crisis: 12-hour daily power cuts devastated manufacturing output by 30%.",
        "Copper prices hit record high as Chile mine strikes and Peruvian political instability cut supply.",
        "Commodity price shock: wheat futures hit 14-year high after Black Sea export corridor collapsed.",
        "Energy transition metals: lithium prices fell 80% from peak as supply from Australia and Chile surged.",
        "Global shipping fuel costs surged 40% as Red Sea rerouting added 6,000 nautical miles to Asia-Europe routes.",
        "OPEC+ agreed to extend production cuts through 2025, keeping 2 million bpd off the market.",
        "US Strategic Petroleum Reserve release of 180 million barrels failed to contain oil price surge.",
        "Europe's energy independence push: €300 billion REPowerEU plan to end Russian fossil fuel dependence.",
        "Critical mineral supply: Congo's state mining company demanded renegotiation of all foreign mining contracts.",
        "Natural gas rationing in Germany as industrial users face mandatory 15% consumption cuts.",
        "Global coal shortage: Indian power plants operated below 30% stockpile levels amid record heatwave.",
        "Cobalt price crash: oversupply from DRC artisanal mining cut prices 50%, hitting Glencore margins.",
        "Water scarcity crisis: Taiwan chip fabs faced production cuts as drought depleted reservoir levels.",
        "Food security: global rice export ban by India affected 40% of world trade in the staple grain.",
    ]
    for text in resource_energy_examples:
        texts.append(text)
        labels.append(CAT2IDX["resource_energy_disruptions"])
        category_counts["resource_energy_disruptions"] += 1

    # Also augment other thin categories from EDGAR
    for cat in ["regulatory_sovereignty_shifts", "sanctions_financial_restrictions"]:
        if category_counts[cat] < MIN_PER_CATEGORY:
            edgar_rows = conn.execute(
                """SELECT mention_text FROM geopolitical_mentions
                   WHERE primary_category = ?
                   AND specificity_score > 15
                   ORDER BY specificity_score DESC
                   LIMIT ?""",
                (cat, MIN_PER_CATEGORY - category_counts[cat]),
            ).fetchall()
            for r in edgar_rows:
                texts.append(r["mention_text"][:512])
                labels.append(CAT2IDX[cat])
                category_counts[cat] += 1

    logger.info("Training data loaded:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {cat:40s} {count:6d}")
    logger.info(f"  {'TOTAL':40s} {len(texts):6d}")

    return texts, labels


def train_model(
    texts: list[str],
    labels: list[int],
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    val_split: float = 0.15,
) -> tuple:
    """Train DistilBERT classifier on the prepared data."""

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Train/val split (stratified)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=val_split, stratify=labels, random_state=42,
    )

    logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

    # Tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(CATEGORIES),
    )
    model.to(device)

    # Datasets
    train_dataset = EventTextDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = EventTextDataset(val_texts, val_labels, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Optimizer with linear warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps,
    )

    # Class weights for imbalanced data
    label_counts = np.bincount(train_labels, minlength=len(CATEGORIES))
    weights = 1.0 / np.maximum(label_counts, 1).astype(float)
    weights = weights / weights.sum() * len(CATEGORIES)  # normalize
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    logger.info(f"Class weights: {dict(zip(CATEGORIES, [f'{w:.2f}' for w in weights]))}")

    # Training loop
    best_val_f1 = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, batch_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch_labels).sum().item()
            total += len(batch_labels)

            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    f"  Epoch {epoch+1}/{epochs} batch {batch_idx+1}/{len(train_loader)}: "
                    f"loss={total_loss/(batch_idx+1):.4f} acc={correct/total:.3f}"
                )

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        # Validation
        val_preds, val_true, val_loss = evaluate(model, val_loader, device, loss_fn)
        val_acc = np.mean(np.array(val_preds) == np.array(val_true))

        # Per-class F1
        report = classification_report(
            val_true, val_preds,
            target_names=CATEGORIES,
            output_dict=True,
            zero_division=0,
        )
        macro_f1 = report["macro avg"]["f1-score"]
        weighted_f1 = report["weighted avg"]["f1-score"]

        logger.info(
            f"Epoch {epoch+1}/{epochs}: "
            f"train_loss={avg_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} "
            f"macro_f1={macro_f1:.3f} weighted_f1={weighted_f1:.3f}"
        )

        # Save best model
        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)

            # Save label mapping
            with open(MODEL_DIR / "label_map.json", "w") as f:
                json.dump({"cat2idx": CAT2IDX, "idx2cat": IDX2CAT}, f, indent=2)

            logger.info(f"  Saved best model (macro_f1={macro_f1:.3f})")

    return model, tokenizer, val_texts, val_labels


def evaluate(model, data_loader, device, loss_fn=None) -> tuple[list, list, float]:
    """Run evaluation, return predictions, true labels, and avg loss."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

            if loss_fn:
                loss = loss_fn(outputs.logits, batch_labels)
                total_loss += loss.item()

    avg_loss = total_loss / max(len(data_loader), 1)
    return all_preds, all_labels, avg_loss


def print_evaluation(val_true, val_preds):
    """Print detailed classification report and confusion matrix."""
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(
        val_true, val_preds,
        target_names=CATEGORIES,
        digits=3,
        zero_division=0,
    ))

    print("CONFUSION MATRIX")
    print("-" * 70)
    cm = confusion_matrix(val_true, val_preds)
    # Print with category abbreviations
    abbrevs = ["trade", "sanct", "armed", "regul", "tech", "resrc", "polit", "instit"]
    header = "          " + "  ".join(f"{a:>6s}" for a in abbrevs)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:6d}" for v in row)
        print(f"{abbrevs[i]:>8s}  {row_str}")
    print()


@click.command()
@click.option("--epochs", default=3, type=int, help="Training epochs")
@click.option("--batch-size", default=16, type=int, help="Batch size")
@click.option("--lr", default=2e-5, type=float, help="Learning rate")
@click.option("--max-per-cat", default=5000, type=int, help="Max samples per category")
@click.option("--max-length", default=256, type=int, help="Max token length")
@click.option("--eval-only", is_flag=True, help="Only evaluate saved model")
@click.option("--model-path", default=None, help="Path to saved model (for eval-only)")
def main(epochs, batch_size, lr, max_per_cat, max_length, eval_only, model_path):
    """Train or evaluate the geopolitical event classifier."""

    if eval_only:
        path = Path(model_path) if model_path else MODEL_DIR
        logger.info(f"Loading model from {path}")
        tokenizer = DistilBertTokenizer.from_pretrained(path)
        model = DistilBertForSequenceClassification.from_pretrained(path)

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model.to(device)

        conn = get_db_connection()
        texts, labels = load_training_data(conn, max_per_cat=max_per_cat)
        conn.close()

        _, val_texts, _, val_labels = train_test_split(
            texts, labels, test_size=0.15, stratify=labels, random_state=42,
        )
        val_dataset = EventTextDataset(val_texts, val_labels, tokenizer, max_length)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        val_preds, val_true, _ = evaluate(model, val_loader, device)
        print_evaluation(val_true, val_preds)
        return

    # Full training pipeline
    conn = get_db_connection()
    texts, labels = load_training_data(conn, max_per_cat=max_per_cat)
    conn.close()

    model, tokenizer, val_texts, val_labels = train_model(
        texts, labels,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        max_length=max_length,
    )

    # Final evaluation on val set
    device = next(model.parameters()).device
    val_dataset = EventTextDataset(val_texts, val_labels, tokenizer, max_length)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    val_preds, val_true, _ = evaluate(model, val_loader, device)
    print_evaluation(val_true, val_preds)

    logger.info(f"Model saved to {MODEL_DIR}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
