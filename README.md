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
- [Learnings](#learnings)

---

## Project Overview

The raw dataset is transactional/time-series-like: multiple rows per `customer_ID` over time (with a statement date column like `S_2`). Most tree models perform best when the data is transformed into a **single row per customer** with engineered features such as:

- Aggregations over time (mean/std/min/max/last, etc.)
- Recent-history features (lags)
- Change features (last − previous)

This repo implements that idea while keeping preprocessing feasible under Colab memory limits.

---

## Repository Structure
 ```
Challenge-AMEX-main/
├─ Data_Preprocessor.ipynb
├─ Final_Model.ipynb
└─ README.md
```

### Notebooks

- **`Data_Preprocessor.ipynb`**
  - Converts large CSV files into **chunked Parquet** with explicit dtype handling.
  - Includes logic to build a **customer-level aggregated dataset** for training.

- **`Final_Model.ipynb`**
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

### 1) Data Cleaning: Denoising & Parquet Conversion

This stage (found in Data_Preprocessor.ipynb) is the "foundation" of the project. We convert the raw 50GB+ CSV data into a format that is small, clean, and fast.

> What is being done:

* Chunking: We read the CSV in blocks of 500,000 rows rather than all at once.

* Denoising (The Rounding Trick): We round all numerical features to 2 decimal places.

* Downcasting: We change how numbers are stored (e.g., changing float64 to float32 and categories to int8).

* Parquet Conversion: The cleaned data is saved as .parquet files instead of .csv.

> Why we do it:

* Memory Efficiency: The raw AMEX dataset is too large for Google Colab's RAM. Downcasting reduces the memory footprint by over 50% without losing important information.

* Removing Noise: The original data contains tiny mathematical fluctuations (e.g., 0.5000001 vs 0.499999). These are "noise" that can confuse the model. Denoising rounds these values, helping the model find real patterns rather than chasing insignificant decimals.

* Speed: Parquet is a "columnar" storage format. Unlike CSVs (which read row-by-row), Parquet allows the computer to skip columns it doesn't need, making data loading up to 10x faster.

---

### 2) Customer-Level Feature Engineering (Aggregation + Lags)

This stage compresses a customer's entire 13-month financial history into a single descriptive "summary card."

> What is being done:

* Aggregation: For every customer, we calculate the Mean (average), Standard Deviation (volatility), Min/Max (extremes), and the Last (most recent) value for every feature.

* Velocity (Diffs): We calculate the difference between the customer's first month and their last month.

* Lags: we look back 1 or 2 months to see very recent changes.

> Why we do it:

* The Model's Perspective: Gradient Boosted Trees (LightGBM/CatBoost) cannot naturally "read" a 13-row history for one person. They need one row per customer. Aggregation provides this summary.

* Capturing Trajectory: A customer with a high balance that is decreasing is very different from a customer with a high balance that is increasing. By calculating Velocity (Last - First), we tell the model not just "where the customer is," but "where they are headed."

* Static vs. Dynamic Health: Summary stats (Mean/Max) show the customer's general financial health, while Lags and Diffs show their immediate financial stress. Combining both gives the model a 360-degree view.

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

Rationale for model choice and evaluation:
LightGBM was primarily used as a fast, strong baseline with cross-validation to validate feature usefulness and generalization under the AMEX evaluation metric. CatBoost was selected as the final model because of its robustness to noisy numerical features, its ability to model complex non-linear interactions, and its stable performance on tabular credit-risk data. Model performance was assessed using the official AMEX metric (Gini + Top-4% capture), ensuring alignment with the competition’s business objective rather than generic accuracy.

Saved artifact:

* `catboost_meta.cbm`

---

### 4) Inference + Submission

Implemented in `Final_Model.ipynb`.

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
   * `Final_Model.ipynb` (aggregates with lags, trains, predicts, exports submission)

You should end with:

* `submission_gold_strategy.csv`

# Learnings

## 1. Why This Challenge Was Useful

Scale: This challenge forced a departure from standard “load everything into RAM” workflows toward engineering-driven machine learning, where data access patterns, storage formats, and memory constraints directly shape modeling decisions. It highlighted the importance of designing pipelines that remain usable under real-world hardware limitations.

Domain Expertise: Working on credit risk data provided concrete insight into how temporal financial behavior translates into default risk. In particular, it showed how risk is often driven less by absolute values than by trends, volatility, and recent changes in customer behavior.

Metric Precision: The competition’s custom metric (Gini combined with Top-4% capture) emphasized the need to optimize for business-oriented objectives rather than generic accuracy. This reinforced the idea that evaluation metrics strongly influence feature design, model choice, and validation strategies.

## 2. Problems & Challenges Encountered

Data Scale (The Memory Wall): With over 50GB of raw data, conventional pandas-based workflows were infeasible. This required adopting a chunked processing mindset, where data is incrementally read, transformed, and persisted, forcing careful control over memory lifecycles and intermediate objects.

Data Noise: The presence of high-precision numerical noise increased storage costs and risked overfitting. Identifying that this noise was non-informative and applying controlled rounding required balancing signal preservation against model stability and efficiency.

Temporal Complexity: Customers exhibited highly variable history lengths, making naive time-series modeling impractical. Designing consistent customer-level summaries that preserved directional behavior (trends and recent changes) without leaking future information was a non-trivial modeling challenge.

## 3. New Skills & Techniques Acquired

Denoising & Compression: The project introduced practical techniques such as numerical rounding and downcasting to reduce dataset size while maintaining predictive signal, demonstrating that data preprocessing can materially affect both feasibility and performance.

Efficient Data Formats: Transitioning from CSV to Parquet highlighted the impact of columnar storage on large-scale ML workflows, enabling selective column loading and significantly faster I/O during feature engineering and inference.

Temporal Feature Design: Creating velocity, lag, and difference features strengthened understanding of how to encode time-dependent behavior into fixed-size representations compatible with tree-based models.

## 4. Why the Final Technical Solution Was Effective

The final solution was effective because it balanced information retention, computational feasibility, and model expressiveness under strict memory constraints. By transforming raw transactional histories into customer-level aggregated and velocity-based features, the pipeline preserved the most predictive temporal signals while remaining compatible with gradient-boosted trees.

The use of Parquet storage and batch-based processing enabled the pipeline to scale to tens of millions of rows without exceeding available RAM. Combined with CatBoost’s robustness to noisy numerical data and ability to model complex non-linear interactions, this design achieved strong performance while remaining operationally realistic. Overall, the solution closely mirrors real-world credit-risk systems, where predictive accuracy must coexist with infrastructure constraints.
