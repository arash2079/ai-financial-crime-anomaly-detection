# Source data

The main dataset is **IBM Transactions for Anti Money Laundering (AML)**,
published on Kaggle by `ealtman2019` and linked from IBM's official repository.
It contains simulated transactions, not real bank customer records.

- Source: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
- Publisher reference: https://github.com/IBM/AML-Data
- Kaggle dataset ID: `2948142`; version: `8`.
- Reported version update: `2025-07-08T14:59:13.28Z`.
- File: `HI-Small_Trans.csv`.
- File size: `475664283` bytes.
- SHA-256: `b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040`.
- Data license: **CDLA-Sharing-1.0**, separate from this project's MIT code license.

Downloaded-file inspection found 5,078,345 transaction rows: 5,073,168 label-0
rows and 5,177 label-1 rows. These are dataset counts, not model results.
Labels refer to simulated laundering activity. The file contains multiple
currencies, duplicate `Account` column names, valid self-transfers and records
that are not globally sorted by timestamp. Import code must account for these
properties rather than assuming the earlier custom simulator's schema.

Most transactions are dated September 1-10, 2022, followed by a sparse tail
through September 18. The initial study uses the main ten-day period rather
than treating the sparse tail as its entire validation/test population.

## Preparation protocol

Fixed before model training:

- `Payment Format` equals `ACH`.
- `Payment Currency` and `Receiving Currency` both equal `US Dollar`.
- Timestamp is at or after `2022/09/01 00:00` and before `2022/09/11 00:00`.
- `Amount Paid` is positive, represented exactly in integer cents.
- Keep self-transfers and original laundering labels; do not oversample,
  undersample, or filter by label.
- Sort by timestamp, preserving source-row order for ties. Composite bank/account
  identifiers avoid accidentally merging accounts from different banks.
- Train: September 1-6; validation: September 7-8; test: September 9-10.

This deliberately bounded subset makes one payment type and currency manageable;
it is not a representative sample of all financial transactions. Exploratory
counts, including label availability, were inspected before freezing the scope;
no model performance was used to choose it. Feature histories cover only the
selected subset. The original file has minute-resolution timestamps without a
timezone; UTC is an internal processing convention rather than verified source
timezone information. No models have been trained in this rebuild yet.

The complete preparation run selected **201,381** transactions, including
**1,419** label-1 records and **113** self-transfers. No zero or negative amounts
remained in this scope. Exclusions were 1,108 out-of-period rows, 3,221,109 rows
with other currency combinations, and 1,654,747 rows with other payment formats.
These disjoint counts reconcile to the 5,078,345 source rows.

| Split | Rows | Label-1 records |
| --- | ---: | ---: |
| Train | 121,962 | 796 |
| Validation | 38,688 | 301 |
| Test | 40,731 | 322 |

These are preparation counts, not classification results.

Keep downloaded files in `data/raw/`, which is ignored by Git. Importer tests use
small manually constructed CSV fixtures, not a replacement training dataset.
The source checksum is mandatory in the preparation CLI. Local
`data/preparation.json` records the filter counts and resulting split sizes.
