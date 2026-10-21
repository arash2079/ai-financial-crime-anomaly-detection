"""Prepare IBM synthetic USD ACH features and fixed splits; no model training."""

import argparse
import csv
import json
from pathlib import Path
from datetime import timezone

from .features import FEATURE_NAMES, build_features, chronological_split
from .data import DEFAULT_INPUT, SOURCE_SHA256, SOURCE_URL, SOURCE_VERSION, load_transactions


def prepare_dataset(input_path=DEFAULT_INPUT, output="data/features.csv", manifest="data/preparation.json"):
    if len({Path(input_path).resolve(), Path(output).resolve(), Path(manifest).resolve()}) != 3:
        raise ValueError("input, output and manifest must be different files")
    records, counts = load_transactions(input_path)
    if not records:
        raise ValueError("no transactions remain in the frozen scope")
    split_names = chronological_split(records)
    features = build_features(records)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    # ID/time and label are metadata columns, never members of FEATURE_NAMES.
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("id", "timestamp", "label", "split", *FEATURE_NAMES))
        for record, split, feature in zip(records, split_names, features):
            writer.writerow((record.id, record.timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                             record.label, split, *(feature[name] for name in FEATURE_NAMES)))
    split_counts = {name: {"rows": 0, "positives": 0} for name in ("train", "validation", "test")}
    for record, split in zip(records, split_names):
        split_counts[split]["rows"] += 1
        split_counts[split]["positives"] += record.label
    report = {
        "source": {"dataset": "IBM Transactions for Anti Money Laundering (synthetic)",
                   "kaggle_dataset_id": 2948142, "version": SOURCE_VERSION,
                   "file": "HI-Small_Trans.csv", "sha256": SOURCE_SHA256, "url": SOURCE_URL},
        "scope": {"start_inclusive": "2022-09-01", "end_exclusive": "2022-09-11",
                  "payment_currency": "US Dollar", "receiving_currency": "US Dollar",
                  "payment_format": "ACH", "amount": "positive Amount Paid; exact integer cents",
                  "sampling": "none; labels do not affect selection", "self_transfers": "retained"},
        "timezone": "Source timezone unknown; UTC attached only as a processing convention.",
        "counts": counts, "splits": split_counts, "feature_names": list(FEATURE_NAMES),
        "protocol": {"train": "2022-09-01 <= t < 2022-09-07",
                     "validation": "2022-09-07 <= t < 2022-09-09",
                     "test": "2022-09-09 <= t < 2022-09-11",
                     "history": "strictly earlier outgoing sender events in [t-24h,t); ties scored before updates; history continues across splits",
                     "cold_start": "count/total/mean/ratio=0; is_new_receiver=1",
                     "features": "USD; no IDs, labels or Patterns.txt predictors"},
    }
    manifest = Path(manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default="data/features.csv")
    parser.add_argument("--manifest", default="data/preparation.json")
    args = parser.parse_args()
    try:
        report = prepare_dataset(args.input, args.output, args.manifest)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Prepared {report['counts']['selected_rows']:,} transactions; splits: {report['splits']}")


if __name__ == "__main__":
    main()
