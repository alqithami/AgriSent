# AgriSent

**AgriSent** is a research repository that supports the manuscript:

> *AI-Driven Agricultural Price Forecasting: Leveraging Consumer Sentiment for Precision Market Intelligence*  
> Saad Alqithami, Musaad Alzahrani

The project investigates whether **consumer sentiment extracted from Arabic/English social-media posts** (Twitter/X) adds predictive signal for **monthly Saudi rice import prices**, when combined with **historical price dynamics** and **local temperature-based climate variables**.

---

## Repository contents

This repository currently contains the following main files (see the repository root):

- `AgriSentRice.py` — model training/evaluation script (baseline implementation).
- `MergeRiceDataWithTwitterSentimentData.py` — data preparation/merging script used to construct the merged panel.
- `merged_climate_sentiment_rice1.csv` — merged monthly panel used by `AgriSentRice.py`.
- `processed_climate_rice_new.csv` — intermediate processed climate dataset.
- `rice_tweets _with_sentiments.xlsx` — tweet-level sentiment export used for aggregation (**see notes on platform terms below**).
- `README.md` — this file.

---

## Data

### 1) Merged monthly panel

The main modeling dataset is:

- **`merged_climate_sentiment_rice1.csv`**

It is a **monthly** dataset with **109 rows** spanning **2015-01 through 2024-01** (inclusive). Key variables include:

- `Date` (monthly timestamp)
- Rice prices:
  - `Basmati_Rice_Price`
  - `Maza_Rice_Price`
- Temperature variables used in the baseline script:
  - `TAVG`, `TMAX`, `TMIN`
- Aggregated sentiment variables (monthly):
  - `average_sentiment`
  - `average_weighted_sentiment`
  - `sum_of_positive_sentiment`, `sum_of_neutral_sentiment`, `sum_of_negative_sentiment`
  - engagement summaries: `sum_of_retweets`, `sum_of_likes`, `sum_of_engagements`, `sum_of_log_scaled_engagement`

**Months with no retained/eligible tweets after filtering may have sentiment aggregates equal to 0.** This keeps a continuous monthly time index for forecasting.

### 2) Tweet-level file

The repository includes:

- **`rice_tweets _with_sentiments.xlsx`**

If this spreadsheet contains **tweet text**, you should verify compliance with Twitter/X developer terms before making the repository broadly public. A common compliant pattern is to release **tweet IDs only** and provide a rehydration script. After acceptance in order to keep this repository permanently public we will consider replacing any raw text with tweet IDs and derived labels/metadata.

---

## Quick start

### A) Create an environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### B) Install dependencies

This repository uses standard scientific Python packages:

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn openpyxl
```

### C) Run the baseline experiment

```bash
python AgriSentRice.py
```

The script reads `merged_climate_sentiment_rice1.csv` from the current working directory and produces:

- console output of split sizes and model metrics
- time-series plots of actual vs predicted values
- residual diagnostic plots (saved to a local directory; see the note below)
- `model_performance_comparison.csv` (written to the working directory)

---

## What `AgriSentRice.py` does (implementation details)

The current version of `AgriSentRice.py` is configured to forecast **Maza rice prices** (`target = 'Maza_Rice_Price'`) using:

### Feature engineering
- Chronological sorting by `Date`.
- Lagged features (note: names reflect the code as implemented):
  - `*_Lag_1 = shift(3)` and `*_Lag_2 = shift(4)`
- Smoothed features:
  - `*_Rolling_3_Lag_1 = rolling(6).mean().shift(3)` (a 6‑month trailing mean, shifted)
- Seasonality:
  - `month` is included as an integer feature (cyclic `month_sin`/`month_cos` are computed but not used in the default feature lists).

After lag/rolling construction, rows with missing values are dropped (`dropna()`), leaving **101 observations** for modeling in the current configuration.

### Train/test split
- A single **chronological** split is used:
  - first 80% of observations for training
  - last 20% for testing

### Models and fixed hyperparameters (current baseline)
- **SARIMAX** (statsmodels):
  - `order=(1,1,1)`
  - `seasonal_order=(1,1,0,12)`
  - exogenous regressors are the engineered feature set
- **Gradient Boosting Regressor** (scikit-learn):
  - `n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42`
- **Random Forest Regressor** (scikit-learn):
  - `n_estimators=200, max_depth=5, random_state=42`
- **Ridge Regression**:
  - `alpha=1.0`
- **Linear Regression** (OLS in scikit-learn)

### Evaluation metrics
- RMSE
- MAE
- out-of-sample R²

---

## Running Basmati vs. Maza

The merged dataset contains both Basmati and Maza price series.

`AgriSentRice.py` is currently configured for **Maza**. To run a comparable experiment for **Basmati**, you can modify:

1. `target = 'Basmati_Rice_Price'`
2. Update `features_to_lag` so it includes `Basmati_Rice_Price` (instead of `Maza_Rice_Price`)
3. Update `features_without_sentiment` and `features_with_sentiment` to use the corresponding lagged Basmati price columns.

**Tip:** Keep the feature engineering logic identical across commodities if you want directly comparable results.

---

## Important note about file paths (residual plots)

The script contains a hard-coded Windows output directory (e.g., `C:\Users\...`) for saving residual plots.

If you are not running on that machine, you should replace that path with a relative path, for example:

```python
output_dir = os.path.join(os.getcwd(), "outputs", "residuals")
```

---

## How to cite

If you use this code or the merged dataset, please cite the associated manuscript. A BibTeX entry can be added here once the paper is accepted/published.

---

## Contact

For questions, please open a GitHub Issue or contact the repository maintainer via GitHub profile.
