"""Seven simple sender-history features with no future information.

Only outgoing events in [current time - 24 hours, current time) are used.
All events at the same instant are scored before any updates. Amount totals
are accumulated in integer cents and converted to USD only for the features.
IDs, labels and account names are not numeric model features.
"""

import math
from collections import Counter, deque
from datetime import timedelta, timezone

from .data import SCOPE_START, TRAIN_END, VALIDATION_END, SCOPE_END, validate_transactions


FEATURE_NAMES = (
    "log_amount",
    "sender_count_24h",
    "sender_total_24h",
    "sender_mean_24h",
    "sender_unique_receivers_24h",
    "is_new_receiver_24h",
    "amount_to_prior_mean_ratio",
)


class _History:
    def __init__(self):
        self.events = deque()
        self.total_cents = 0
        self.receivers = Counter()

    def expire(self, cutoff):
        while self.events and self.events[0][0] < cutoff:
            _, amount, receiver = self.events.popleft()
            self.total_cents -= amount
            self.receivers[receiver] -= 1
            if self.receivers[receiver] == 0:
                del self.receivers[receiver]

    def add(self, record, timestamp):
        self.events.append((timestamp, record.amount_cents, record.receiver))
        self.total_cents += record.amount_cents
        self.receivers[record.receiver] += 1


def build_features(records):
    """Return one feature dictionary per input record, preserving its order.

    The input must already be chronologically sorted and have unique IDs.
    Empty input returns []. History is local to this call; records are not
    mutated. log_amount is log1p(current USD amount). Other amount features
    are USD, not logarithms. The ratio is current amount / prior mean;
    cold-start count, total, mean and ratio are zero, while new receiver is 1.
    """
    records = list(records)
    validate_transactions(records)
    histories = {}
    features = []
    index = 0
    while index < len(records):
        timestamp = records[index].timestamp.astimezone(timezone.utc)
        end = index + 1
        while end < len(records) and records[end].timestamp.astimezone(timezone.utc) == timestamp:
            end += 1

        for record in records[index:end]:
            history = histories.setdefault(record.sender, _History())
            history.expire(timestamp - timedelta(hours=24))
            count = len(history.events)
            total = history.total_cents / 100
            mean = total / count if count else 0.0
            amount = record.amount_cents / 100
            features.append({
                "log_amount": math.log1p(amount),
                "sender_count_24h": count,
                "sender_total_24h": total,
                "sender_mean_24h": mean,
                "sender_unique_receivers_24h": len(history.receivers),
                "is_new_receiver_24h": int(record.receiver not in history.receivers),
                "amount_to_prior_mean_ratio": amount / mean if count else 0.0,
            })
        for record in records[index:end]:
            histories[record.sender].add(record, timestamp)
        index = end
    return features


def chronological_split(records):
    """Use fixed Sep 1-6 / Sep 7-8 / Sep 9-10, 2022 splits.

    UTC is a processing convention. Reject dates outside the frozen scope.
    History is continuous across splits; ties cannot straddle a boundary.
    """
    records = list(records)
    validate_transactions(records)
    splits = []
    for record in records:
        timestamp = record.timestamp.astimezone(timezone.utc)
        if not SCOPE_START <= timestamp < SCOPE_END:
            raise ValueError("record is outside the frozen 2022-09-01 to 2022-09-11 scope")
        splits.append("train" if timestamp < TRAIN_END else "validation" if timestamp < VALIDATION_END else "test")
    return splits
