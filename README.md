# AMEX Challenge (Default Prediction) — End-to-End Pipeline

This repository contains a **notebook-based** solution for the American Express default prediction challenge, with a strong focus on **memory-efficient preprocessing**, **feature aggregation at customer level**, and **gradient-boosted tree modeling** (LightGBM + CatBoost).

The workflow is designed primarily for **Google Colab + Google Drive**, where RAM constraints require chunked processing and intermediate checkpoints.

---

## Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Data Assumptions](#data-assumptions)
- [Environment Setup](#environment-setup)
- [Pipeline Summary](#pipeline-summary)
  - [1) CSV ➜ Parquet (Chunked, Memory-Optimized)](#1-csv--parquet-chunked-memory-optimized)
  - [2) Customer-Level Feature Engineering (Aggregation + Lags)](#2-customer-level-feature-engineering-aggregation--lags)
  - [3) Training (LightGBM CV + CatBoost Final)](#3-training-lightgbm-cv--catboost-final)
  - [4) Inference + Submission](#4-inference--submission)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Notes & Known Limitations](#notes--known-limitations)

---

## Project Overview

The raw dataset is transactional/time-series-like: multiple rows per `customer_ID` over time (with a statement date column like `S_2`). Most tree models perform best when the data is transformed into a **single row per customer** with engineered features such as:

- Aggregations over time (mean/std/min/max/last, etc.)
- Recent-history features (lags)
- Change features (last − previous)

This repo implements that idea while keeping preprocessing feasible under Colab memory limits.

---

## Repository Structure

Challenge-AMEX-main/
├─ Data_Preprocessor.ipynb
├─ Another_copy_of_Test_Meryem_Model.ipynb
└─ README.md

### Notebooks

- **`Data_Preprocessor.ipynb`**
  - Converts large CSV files into **chunked Parquet** with explicit dtype handling.
  - Includes logic to build a **customer-level aggregated dataset** for training.

- **`Another_copy_of_Test_Meryem_Model.ipynb`**
  - Uses **Polars** for fast lazy scanning and aggregation.
  - Builds lag/diff features for selected “key features”.
  - Trains LightGBM (CV) and a final CatBoost model.
  - Runs batch inference on test Parquet and produces a submission file.

---

## Data Assumptions

The notebooks expect the following files to exist in a single base directory (typically on Google Drive):

- `train_data.csv`
- `test_data.csv`
- `train_labels.csv`

The pipeline then creates:
- Chunked Parquet folders: `train_parquet/`, `test_parquet/`
- Aggregated tables: `train_aggregated.parquet` / `train_aggregated_v1.parquet`
- Test batches: `test_batch_*.parquet`
- Submission: `submission_gold_strategy.csv`

---

## Environment Setup

This project is designed around **Google Colab**.

In Colab, install dependencies (the training notebook already includes this):

```bash
pip install polars catboost lightgbm
````

> The notebook also imports `xgboost`, but the core pipeline shown relies on LightGBM + CatBoost.

---

## Pipeline Summary

### 1) CSV ➜ Parquet (Chunked, Memory-Optimized)

Implemented in `Data_Preprocessor.ipynb`.

Key ideas:

* Read CSV using `pandas.read_csv(..., chunksize=...)` to avoid loading the full dataset into RAM.
* Apply **type normalization**:

  * Convert date column(s) to proper datetime
  * Map the only two string categorical fields consistently:

    * `D_63` mapped via a fixed dictionary
    * `D_64` mapped via a fixed dictionary
  * Ensure other categorical columns are numeric and cast to compact integer dtypes (`int8`)
  * Downcast numeric floats to `float32` to reduce memory

Outputs:

* `train_parquet/part_00000.parquet`, `part_00001.parquet`, ...
* `test_parquet/part_00000.parquet`, `part_00001.parquet`, ...

Why this matters:

* Parquet + dtype control dramatically reduces RAM usage and speeds up later scanning/aggregation.

---

### 2) Customer-Level Feature Engineering (Aggregation + Lags)

Implemented across both notebooks (with a more scalable version using Polars in the training notebook).

**Aggregation concept (1 row per customer):**

* For numeric features: compute a set of summary stats (the notebooks reference a plan such as mean/std/min/max/last).
* For categorical features: retain last/most recent values (common in AMEX solutions).

**Lag + diff features (Polars-based)**
In `Another_copy_of_Test_Meryem_Model.ipynb`, additional temporal features are added for selected key columns:

Key feature set (as coded):

* `P_2`, `B_1`, `B_2`, `B_9`, `D_39`, `S_3`

For each of these, the notebook computes:

* `*_lag_1`: previous statement value
* `*_lag_2`: value two steps back
* `*_diff_last_lag1`: last − lag_1 (a short-term change indicator)

Implementation notes:

* Uses **Polars lazy API** (`scan_parquet`) + `group_by("customer_ID").agg(exprs)` for speed.
* Handles short histories gracefully (lags can be null if not enough rows exist).

---

### 3) Training (LightGBM CV + CatBoost Final)

Implemented in `Another_copy_of_Test_Meryem_Model.ipynb`.

High-level flow:

1. Load aggregated training table from Parquet (customer-level).
2. Build feature list excluding identifiers and label columns (the notebook excludes: `customer_ID`, `target`, and `S_2`).
3. Train:

   * **LightGBM** using **Stratified K-Fold** CV (models retained in memory for later inference)
   * A **CatBoost** model trained on the full training data as the “final model”

The notebook indicates a “meta” approach where LightGBM helps inform the final CatBoost strategy (as described in comments). Operationally:

* LightGBM models are kept for generating auxiliary predictions.
* CatBoost is trained as the final predictor and saved to disk.

Saved artifact:

* `catboost_meta.cbm`

---

### 4) Inference + Submission

Implemented in `Another_copy_of_Test_Meryem_Model.ipynb`.

To avoid RAM issues:

* The test set is processed in **batches** (`test_batch_*.parquet`), typically grouping a small number of chunk files at a time.
* For each batch:

  * Load the batch parquet
  * Compute predictions
  * Append to a list of partial submission frames

Final steps:

* Concatenate all batch predictions
* Deduplicate by `customer_ID` using mean prediction (groupby aggregation)
* Save submission CSV:

  * `submission_gold_strategy.csv`

---

## Configuration

Both notebooks hard-code a Drive path similar to:

```python
base_path = "/content/drive/MyDrive/AMEX Challenge"
# or
BASE_DIR = "/content/drive/MyDrive/AMEX Challenge"
```

To run locally or in a different Drive folder:

1. Update `base_path` / `BASE_DIR` in both notebooks.
2. Ensure input files exist in that directory:

   * `train_data.csv`, `test_data.csv`, `train_labels.csv`

---

## Outputs

Typical outputs produced by the full pipeline:

* **Intermediate**

  * `train_parquet/` (chunked)
  * `test_parquet/` (chunked)
  * `train_aggregated.parquet` (customer-level with labels)
  * `test_batch_*.parquet` (customer-level batches)

* **Model**

  * `catboost_meta.cbm`

* **Submission**

  * `submission_gold_strategy.csv` with columns:

    * `customer_ID`
    * `prediction`

---

## Notes & Known Limitations

* **Notebook-first repo**: there is no packaged CLI or python module; execution is notebook-driven.
* **Paths are hard-coded**: intended for Colab; update variables to run elsewhere.
* **Some cells are marked as heavy / not meant to run blindly** (e.g., full aggregation steps). Expect long runtimes on full data.
* **Reproducibility**: random seeds are set in places (e.g., `random_state=42`), but full determinism depends on environment and library versions.
* **Memory management**: the notebooks explicitly `del` objects and call `gc.collect()` to stay within Colab RAM limits.

---

## Quick Start (Colab)

1. Upload dataset CSVs and `train_labels.csv` to your Drive folder.
2. Open and run:

   * `Data_Preprocessor.ipynb` (creates chunked Parquet)
   * `Another_copy_of_Test_Meryem_Model.ipynb` (aggregates with lags, trains, predicts, exports submission)

You should end with:

* `submission_gold_strategy.csv`

