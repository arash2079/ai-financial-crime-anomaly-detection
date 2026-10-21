"""Small fixture-based checks for the IBM importer and causal features."""

import csv
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import io
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.data import (
    HEADER, SOURCE_SHA256, SOURCE_VERSION, Transaction, load_transactions, sha256_file,
)
from src.features import FEATURE_NAMES, build_features, chronological_split
from src.download_data import download_data


START = datetime(2022, 9, 1, tzinfo=timezone.utc)
IBM_HEADER = (
    "Timestamp", "From Bank", "Account", "To Bank", "Account", "Amount Received",
    "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering",
)


def transfer(identifier, seconds=0, sender="A", receiver="B", cents=200, label=0):
    return Transaction(
        id=identifier, timestamp=START + timedelta(seconds=seconds), sender=sender,
        receiver=receiver, amount_cents=cents, label=label,
    )


def source_row(timestamp="2022/09/01 00:00", from_bank="1", sender="A", to_bank="2",
               receiver="B", received="2.00", receiving_currency="US Dollar", paid="2.00",
               payment_currency="US Dollar", payment_format="ACH", label="0"):
    # Positions matter: the official header has two columns named Account.
    return [timestamp, from_bank, sender, to_bank, receiver, received, receiving_currency,
            paid, payment_currency, payment_format, label]


class FeatureTests(unittest.TestCase):
    def test_hand_computed_sender_history_and_cold_start(self):
        records = [
            transfer("first", cents=200),
            transfer("incoming-only", seconds=1800, sender="B", receiver="A", cents=9900),
            transfer("second", seconds=3600, receiver="C", cents=300),
            transfer("third", seconds=7200, receiver="B", cents=1000),
        ]
        original = list(records)
        expected = [
            (math.log1p(2), 0, 0, 0, 0, 1, 0),
            (math.log1p(99), 0, 0, 0, 0, 1, 0),
            (math.log1p(3), 1, 2, 2, 1, 1, 1.5),
            (math.log1p(10), 2, 5, 2.5, 2, 0, 4),
        ]
        self.assertEqual(tuple(FEATURE_NAMES), (
            "log_amount", "sender_count_24h", "sender_total_24h", "sender_mean_24h",
            "sender_unique_receivers_24h", "is_new_receiver_24h", "amount_to_prior_mean_ratio",
        ))
        features = build_features(records)
        self.assertEqual(len(features), len(records))
        for actual, values in zip(features, expected):
            self.assertEqual(set(actual), set(FEATURE_NAMES))
            for name, value in zip(FEATURE_NAMES, values):
                with self.subTest(feature=name, expected=value):
                    self.assertAlmostEqual(actual[name], value, places=12)
        self.assertEqual(records, original)

    def test_exact_24_hour_boundary_and_cent_preservation(self):
        records = [
            transfer("old", cents=1_660_000_000_000),
            transfer("one-second-later", seconds=1, receiver="C", cents=1),
            transfer("boundary", seconds=86400, receiver="D", cents=400),
            transfer("after-boundary", seconds=86401, receiver="E", cents=500),
        ]
        features = build_features(records)
        self.assertEqual(features[2]["sender_count_24h"], 2)
        self.assertEqual(features[2]["sender_total_24h"], 16_600_000_000.01)
        self.assertEqual(features[3]["sender_count_24h"], 2)
        self.assertEqual(features[3]["sender_total_24h"], 4.01)

    def test_equal_instants_are_scored_before_history_updates(self):
        same_instant = (START + timedelta(hours=1)).astimezone(timezone(timedelta(hours=-4)))
        records = [
            transfer("prior", receiver="Z", cents=200),
            transfer("tie-one", seconds=3600, receiver="B", cents=300),
            replace(transfer("tie-two", receiver="C", cents=500), timestamp=same_instant),
            transfer("following", seconds=3601, receiver="B", cents=700),
        ]
        features = build_features(records)
        for index in (1, 2):
            self.assertEqual(features[index]["sender_count_24h"], 1)
            self.assertEqual(features[index]["sender_total_24h"], 2)
        self.assertEqual(features[3]["sender_count_24h"], 3)
        self.assertEqual(features[3]["sender_total_24h"], 10)
        self.assertEqual(features[3]["is_new_receiver_24h"], 0)

    def test_appending_future_transactions_does_not_change_past_features(self):
        prefix = [transfer("p0"), transfer("p1", seconds=60, receiver="C")]
        future = [transfer("f0", seconds=120, cents=100_000),
                  transfer("f1", seconds=86400, sender="B", receiver="A")]
        self.assertEqual(build_features(prefix), build_features(prefix + future)[:2])

    def test_labels_and_transaction_ids_do_not_enter_features(self):
        records = [transfer("x"), transfer("y", seconds=60, label=1)]
        changed = [replace(r, id="other-" + r.id, label=1 - r.label) for r in records]
        self.assertEqual(build_features(records), build_features(changed))

    def test_consistent_account_renaming_preserves_features(self):
        records = [transfer("x"), transfer("y", seconds=60, sender="B", receiver="A"),
                   transfer("z", seconds=120, receiver="C")]
        renamed = [replace(r, sender="renamed-" + r.sender, receiver="renamed-" + r.receiver)
                   for r in records]
        self.assertEqual(build_features(records), build_features(renamed))

    def test_unsorted_records_and_duplicate_ids_are_rejected(self):
        first, later = transfer("x"), transfer("y", seconds=1)
        for records in ([later, first], [first, replace(later, id="x")]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                build_features(records)

    def test_fixed_calendar_splits_and_timestamp_ties(self):
        offsets = [0, 6 * 86400 - 1, 6 * 86400, 6 * 86400,
                   8 * 86400 - 1, 8 * 86400, 10 * 86400 - 1]
        records = [transfer(str(i), seconds=offset) for i, offset in enumerate(offsets)]
        self.assertEqual(chronological_split(records),
                         ["train", "train", "validation", "validation", "validation", "test", "test"])
        # Assignment uses fixed dates, not the span of the supplied batch.
        self.assertEqual(chronological_split([records[-1]]), ["test"])
        for offset in (-1, 10 * 86400):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                chronological_split([transfer("outside", seconds=offset)])


class ImportTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "fixture.csv"

    def write_fixture(self, rows, header=IBM_HEADER):
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        return self.path

    def load_fixture(self, rows, header=IBM_HEADER):
        return load_transactions(self.write_fixture(rows, header), verify_hash=False)

    def test_exact_ibm_header_and_duplicate_account_columns(self):
        self.assertEqual(tuple(HEADER), IBM_HEADER)
        records, _ = self.load_fixture([source_row(sender="source", receiver="target")])
        self.assertEqual(records[0].sender, "1:source")
        self.assertEqual(records[0].receiver, "2:target")
        invalid = list(IBM_HEADER)
        invalid[4] = "Account.1"
        with self.assertRaises(ValueError):
            self.load_fixture([source_row()], header=invalid)

    def test_filter_bounds_and_disjoint_exclusion_counts(self):
        rows = [
            source_row(),
            source_row(timestamp="2022/09/10 23:59", label="1"),
            source_row(timestamp="2022/08/31 23:59"),
            source_row(timestamp="2022/09/11 00:00"),
            source_row(receiving_currency="Euro"),
            source_row(payment_currency="Euro"),
            source_row(payment_format="Wire"),
            source_row(paid="0.00"),
            source_row(paid="-1.00"),
        ]
        records, counts = self.load_fixture(rows)
        self.assertEqual(len(records), 2)
        expected = {
            "source_rows": 9, "selected_rows": 2, "selected_positives": 1,
            "excluded_date": 2, "excluded_currency": 2, "excluded_format": 1,
            "excluded_zero_amount": 1, "excluded_negative_amount": 1, "self_transfers": 0,
        }
        for key, value in expected.items():
            with self.subTest(count=key):
                self.assertEqual(counts[key], value)
        self.assertEqual(sum(counts[key] for key in expected if key.startswith("excluded_"))
                         + counts["selected_rows"], counts["source_rows"])

    def test_sort_is_stable_and_ids_keep_original_data_row_positions(self):
        records, _ = self.load_fixture([
            source_row(timestamp="2022/09/03 00:00"),
            source_row(payment_format="Wire"),
            source_row(sender="first-tie"),
            source_row(sender="second-tie"),
        ])
        self.assertEqual([r.id for r in records], ["source-row-3", "source-row-4", "source-row-1"])
        self.assertEqual([r.sender for r in records[:2]], ["1:first-tie", "1:second-tie"])
        self.assertTrue(all(r.timestamp.utcoffset() == timedelta(0) for r in records))

    def test_bank_qualified_identity_and_self_transfers_are_preserved(self):
        records, counts = self.load_fixture([
            source_row(sender="shared", receiver="shared", to_bank="2"),
            source_row(sender="shared", receiver="shared", to_bank="1"),
        ])
        self.assertNotEqual(records[0].sender, records[0].receiver)
        self.assertEqual(records[1].sender, records[1].receiver)
        self.assertEqual(counts["self_transfers"], 1)
        self.assertEqual(build_features(records)[1]["sender_count_24h"], 0)

    def test_paid_amount_uses_exact_decimal_cents_and_has_no_obsolete_cap(self):
        records, _ = self.load_fixture([
            source_row(paid="0.01", received="900.00"),
            source_row(paid="1.23"),
            source_row(paid="16600000000.01"),
            source_row(paid="1e3"),
        ])
        self.assertEqual([r.amount_cents for r in records], [1, 123, 1_660_000_000_001, 100_000])
        self.assertTrue(all(type(r.amount_cents) is int for r in records))

    def test_subcent_and_nonfinite_source_amounts_are_rejected(self):
        for value in ("0.001", "1.231", "NaN", "Infinity", "", "not-money"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load_fixture([source_row(paid=value)])

    def test_malformed_rows_dates_labels_and_accounts_are_rejected(self):
        cases = [source_row()[:-1], source_row() + ["extra"],
                 source_row(timestamp="not-a-date"), source_row(label="2"),
                 source_row(from_bank=""), source_row(receiver="")]
        for row in cases:
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.load_fixture([row])

    def test_invalid_transactions_are_rejected_but_large_self_transfer_is_valid(self):
        valid = dict(id="x", timestamp=START, sender="A", receiver="B", amount_cents=100, label=0)
        changes = [{"id": ""}, {"sender": ""}, {"receiver": ""},
                   {"timestamp": START.replace(tzinfo=None)}, {"amount_cents": 0},
                   {"amount_cents": -1}, {"amount_cents": 1.5}, {"amount_cents": True},
                   {"label": 2}, {"label": True}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                Transaction(**(valid | change))
        record = Transaction(**(valid | {"receiver": "A", "amount_cents": 1_660_000_000_001}))
        self.assertEqual(record.sender, record.receiver)

    def test_default_checksum_verification_rejects_unpinned_file(self):
        self.write_fixture([source_row()])
        actual_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.assertEqual(sha256_file(self.path), actual_hash)
        self.assertEqual(str(SOURCE_VERSION), "8")
        self.assertNotEqual(actual_hash, SOURCE_SHA256)
        with self.assertRaises(ValueError):
            load_transactions(self.path)
        records, _ = load_transactions(self.path, verify_hash=False)
        self.assertEqual(len(records), 1)

    def test_downloader_publishes_only_verified_bytes_and_preserves_bad_cache(self):
        with patch("src.download_data.urllib.request.urlopen", return_value=io.BytesIO(b"corrupt")):
            with self.assertRaises(ValueError):
                download_data(self.path)
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.path.parent.iterdir()), [], "Failed downloads must clean up candidates.")

        content = b"small fixture used only to test downloader integrity\n"
        expected_hash = hashlib.sha256(content).hexdigest()
        with patch("src.download_data.SOURCE_SHA256", expected_hash):
            with patch("src.download_data.urllib.request.urlopen", return_value=io.BytesIO(content)):
                self.assertEqual(download_data(self.path), self.path)
            self.assertEqual(self.path.read_bytes(), content)
            with patch("src.download_data.urllib.request.urlopen", side_effect=AssertionError("No network expected")):
                self.assertEqual(download_data(self.path), self.path)

        # Under the actual source checksum this fixture is a bad existing cache.
        with patch("src.download_data.urllib.request.urlopen", side_effect=AssertionError("No network expected")):
            with self.assertRaises(ValueError):
                download_data(self.path)
        self.assertEqual(self.path.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
