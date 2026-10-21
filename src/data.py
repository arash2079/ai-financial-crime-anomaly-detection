"""Load IBM's synthetic USD ACH data; UTC is a processing convention only.

The source does not specify a timezone. No real banking data is used.
"""

import csv
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

SOURCE_VERSION = 8
SOURCE_SHA256 = "b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040"
SOURCE_URL = ("https://www.kaggle.com/api/v1/datasets/download/"
              "ealtman2019/ibm-transactions-for-anti-money-laundering-aml/"
              "HI-Small_Trans.csv?datasetVersionNumber=8")
DEFAULT_INPUT = "data/raw/HI-Small_Trans.csv"
HEADER = ["Timestamp", "From Bank", "Account", "To Bank", "Account", "Amount Received",
          "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"]
SCOPE_START = datetime(2022, 9, 1, tzinfo=timezone.utc)
TRAIN_END = datetime(2022, 9, 7, tzinfo=timezone.utc)
VALIDATION_END = datetime(2022, 9, 9, tzinfo=timezone.utc)
SCOPE_END = datetime(2022, 9, 11, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Transaction:
    id: str
    timestamp: datetime
    sender: str
    receiver: str
    amount_cents: int
    label: int

    def __post_init__(self):
        for name in ("id", "sender", "receiver"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or len(value) > 128:
                raise ValueError(f"{name} must be a non-empty string up to 128 characters")
        if not isinstance(self.timestamp, datetime) or self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be a timezone-aware datetime")
        if type(self.amount_cents) is not int or self.amount_cents <= 0:
            raise ValueError("amount_cents must be a positive integer")
        if type(self.label) is not int or self.label not in (0, 1):
            raise ValueError("label must be the integer 0 or 1")


def validate_transactions(records):
    seen, previous = set(), None
    for record in records:
        if not isinstance(record, Transaction):
            raise ValueError("records must contain Transaction objects")
        timestamp = record.timestamp.astimezone(timezone.utc)
        if previous is not None and timestamp < previous:
            raise ValueError("records must be sorted chronologically")
        if record.id in seen:
            raise ValueError("transaction IDs must be unique")
        seen.add(record.id)
        previous = timestamp


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_cents(text):
    try:
        value = Decimal(text)
        if not value.is_finite():
            raise ValueError("amount must be finite")
        numerator, denominator = value.as_integer_ratio()
        cents, remainder = divmod(numerator * 100, denominator)
        if remainder:
            raise ValueError("amount must have at most two decimal places")
        return cents
    except InvalidOperation as error:
        raise ValueError("amount must be a decimal number") from error


def load_transactions(path, verify_hash=True):
    """Return (sorted records, disjoint exclusion counts) in the frozen scope.

    verify_hash=False is for small test fixtures only. Real preparation always
    verifies the pinned raw file. Source data-row numbers are stable IDs.
    Self-transfers are retained; Patterns.txt is never loaded.
    """
    if verify_hash and sha256_file(path) != SOURCE_SHA256:
        raise ValueError("source SHA-256 does not match pinned dataset version 8")
    counts = dict.fromkeys(("source_rows", "excluded_date", "excluded_currency", "excluded_format",
                           "excluded_zero_amount", "excluded_negative_amount", "selected_rows",
                           "selected_positives", "self_transfers"), 0)
    records = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != HEADER:
            raise ValueError("unexpected CSV header; both positional Account columns are required")
        for source_row, row in enumerate(reader, 1):
            counts["source_rows"] += 1
            if len(row) != len(HEADER):
                raise ValueError(f"source row {source_row}: expected 11 columns")
            try:
                timestamp = datetime.fromisoformat(row[0].replace("/", "-"))
                if timestamp.tzinfo is not None:
                    raise ValueError("source timestamp must be timezone-naive")
                timestamp = timestamp.replace(tzinfo=timezone.utc)
                if not SCOPE_START <= timestamp < SCOPE_END:
                    counts["excluded_date"] += 1
                    continue
                if row[6] != "US Dollar" or row[8] != "US Dollar":
                    counts["excluded_currency"] += 1
                    continue
                if row[9] != "ACH":
                    counts["excluded_format"] += 1
                    continue
                amount_cents = _parse_cents(row[7])
                if amount_cents <= 0:
                    counts["excluded_zero_amount" if amount_cents == 0 else "excluded_negative_amount"] += 1
                    continue
                if not all(row[index].strip() for index in (1, 2, 3, 4)):
                    raise ValueError("bank and account identifiers must be non-empty")
                if row[10] not in ("0", "1"):
                    raise ValueError("Is Laundering must be 0 or 1")
                record = Transaction(f"source-row-{source_row}", timestamp,
                                     f"{row[1]}:{row[2]}", f"{row[3]}:{row[4]}",
                                     amount_cents, int(row[10]))
                records.append(record)
                counts["selected_positives"] += record.label
                counts["self_transfers"] += int(record.sender == record.receiver)
            except ValueError as error:
                raise ValueError(f"source row {source_row}: {error}") from error
    records.sort(key=lambda record: record.timestamp)
    counts["selected_rows"] = len(records)
    return records, counts
