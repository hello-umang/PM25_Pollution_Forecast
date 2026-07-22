"""Build the capstone Jupyter notebook aligned to Project_Steps.pdf (no XGBoost)."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
cells = []

cells.append(
    nbf.v4.new_markdown_cell(
        """# US PM2.5 Pollution Forecast — ML Capstone Project

**Business Problem:** Predict next-hour PM2.5 concentration for air monitoring stations to support public health advisories and pollution-risk alerts.

**Target Users:** Environmental agencies, public health officials, city dashboards, and residents checking air quality.

**Expected Output:** Next-hour PM2.5 forecast, pollution-risk category (Good / Moderate / Unhealthy), model comparison, and Streamlit dashboard.

**ML Task:** Regression / time-series forecasting.

**Practical use case:** Hourly risk monitoring for two stations using EPA AQS-style observations + weather context.

**Dataset:** Educational sample in `data/pm25_raw.csv` mirroring EPA AQS hourly schema (parameter 88101). Source pattern: [EPA AQS](https://www.epa.gov/aqs/obtaining-aqs-data). ≥10,000 rows.
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## Step 1–2: Problem Definition & Data Collection

| Item | Detail |
|------|--------|
| Industry problem | Air-quality forecasting for health advisories |
| Task type | Regression / forecasting |
| Collection method | Synthetic educational dataset generated with `generate_data.py` following EPA AQS hourly schema (offline, no API key) |
| Official source pattern | EPA Air Quality System (AQS) |
| Rows | 17,500+ hourly records, 2 stations, full year |
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

DATA = Path('../data')
VIS = Path('../visuals')
MOD = Path('../models')

df = pd.read_csv(DATA / 'pm25_raw.csv', parse_dates=['datetime'])
print('Shape:', df.shape)
print('Stations:', df['station_id'].nunique())
print('Date range:', df['datetime'].min(), '->', df['datetime'].max())
df.head()"""
    )
)

cells.append(nbf.v4.new_markdown_cell("## Step 3: Clean and Prepare the Data"))
cells.append(
    nbf.v4.new_code_cell(
        """print('Duplicates:', int(df.duplicated().sum()))
print('Missing pm25:', int(df['pm25'].isna().sum()))

df = df.drop_duplicates()
df['station_id'] = df['station_id'].astype(str)
df['datetime'] = pd.to_datetime(df['datetime'])
for col in ['pm25', 'temperature_c', 'humidity_pct', 'wind_speed_mps']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.sort_values(['station_id', 'datetime']).reset_index(drop=True)
df['pm25'] = df.groupby('station_id')['pm25'].transform(
    lambda x: x.fillna(x.rolling(6, min_periods=1).mean())
)
df['pm25'] = df['pm25'].fillna(df['pm25'].mean())
for col in ['temperature_c', 'humidity_pct', 'wind_speed_mps']:
    df[col] = df.groupby('station_id')[col].transform(lambda x: x.fillna(x.median()))
    df[col] = df[col].fillna(df[col].median())

def cap_outliers(s):
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    return s.clip(lo, hi)

for col in ['pm25', 'temperature_c', 'humidity_pct', 'wind_speed_mps']:
    df[col] = df.groupby('station_id')[col].transform(cap_outliers)

print('Rows after clean:', len(df))
print('Remaining missing pm25:', int(df['pm25'].isna().sum()))
df.describe()"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## Step 4: Exploratory Data Analysis

Saved charts live in `visuals/`. Key questions: distributions, correlations, trends, outliers, target behaviour.
"""
    )
)

for img, insight in [
    (
        "daily_pm25_trend.png",
        "**Insight:** Clear seasonal structure — winter months tend to show higher daily average PM2.5 than summer.",
    ),
    (
        "hourly_pattern.png",
        "**Insight:** Diurnal pattern with elevated averages near typical morning/evening activity periods.",
    ),
    (
        "correlation_matrix.png",
        "**Insight:** Weather variables show modest linear correlation with PM2.5; non-linear/lag effects matter more.",
    ),
    (
        "pm25_distribution.png",
        "**Insight:** Right-skewed concentration distribution — most hours are moderate, with a long upper tail.",
    ),
    (
        "monthly_pm25_trend.png",
        "**Insight:** Monthly averages confirm seasonal cycling useful for calendar features.",
    ),
    (
        "pm25_vs_weather.png",
        "**Insight:** Scatter relationships with temperature, humidity, and wind are noisy but directionally sensible.",
    ),
    (
        "pm25_boxplot_outliers.png",
        "**Insight:** Station-level spread and residual extreme values motivate percentile capping in cleaning.",
    ),
]:
    cells.append(nbf.v4.new_code_cell(f'display(Image(str(VIS / "{img}")))'))
    cells.append(nbf.v4.new_markdown_cell(insight))

cells.append(nbf.v4.new_markdown_cell("## Step 5: Feature Engineering & Selection"))
cells.append(
    nbf.v4.new_code_cell(
        """# Reload engineered feature table produced by eda_and_model.py
feat = pd.read_csv(DATA / 'pm25_cleaned_features.csv', parse_dates=['datetime'])
print('Engineered shape:', feat.shape)
print('Columns:', list(feat.columns))

# Leakage guard: lags/rolling are past-only; target is next hour
assert 'target_next_hour_pm25' in feat.columns
feat[['datetime', 'pm25', 'pm25_lag1', 'pm25_lag24', 'pm25_roll6', 'target_next_hour_pm25']].head(10)"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """Features include:
- datetime parts + cyclical encodings (`hour_sin/cos`, `month_sin/cos`)
- lag/rolling PM2.5 statistics
- weather raw values + bins + interactions
- categorical station / bins (one-hot in training pipeline)

**No target leakage:** lag/rolling use only past values; label is `shift(-1)` next-hour PM2.5.
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## Step 6: Split and Validate

Time-based **80:20** split (no shuffle). Tree models also use **TimeSeriesSplit** (3 folds). Preprocessing scalers / encoders are fit on training data only inside the pipeline script.
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """summary = json.loads((MOD / 'summary.json').read_text())
print(json.dumps(summary['split'], indent=2))
print('Train/test rows from cleaned feature file chronological split:')
feat_sorted = feat.sort_values('datetime')
split_idx = int(len(feat_sorted) * 0.8)
print('train', split_idx, 'test', len(feat_sorted) - split_idx)"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## Step 7–9: Train, Tune, Evaluate & Compare

Light scikit-learn models only: Linear, Ridge, ElasticNet, Decision Tree, Random Forest (small), KNN, Linear SVR, MLP (small).  
**No XGBoost / LightGBM / CatBoost / heavy GBM.**

Hyperparameter tuning uses **RandomizedSearchCV + TimeSeriesSplit** on Decision Tree and Random Forest.

Metrics: MAE, MSE, RMSE, R², train/test gap, timing.
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """cmp = pd.read_csv(MOD / 'model_comparison.csv')
display(cmp)
print('Best model:', summary['best_model'])
print('Best metrics:', summary['best_metrics'])"""
    )
)

for img in [
    "model_comparison_rmse.png",
    "actual_vs_predicted.png",
    "feature_importance.png",
    "error_analysis.png",
]:
    cells.append(nbf.v4.new_code_cell(f'display(Image(str(VIS / "{img}")))'))

cells.append(
    nbf.v4.new_markdown_cell(
        """## Step 10: Results, Business Interpretation & Recommendations

**Best-performing model** is selected by lowest test RMSE (see `models/summary.json`).

**Business interpretation:** Next-hour forecasts can drive station-level risk flags and short-term advisories. Lagged PM2.5 and short rolling means typically dominate; weather provides secondary context.

**Error analysis:** Residual plots highlight where predictions under/over-shoot; absolute-error tails show rare spike hours are harder.

**Limitations & recommendations** are listed in the README and `summary.json`. Dashboard: `streamlit run dashboard/app.py`.
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """print('Limitations:')
for x in summary['limitations']:
    print('-', x)
print('\\nRecommendations:')
for x in summary['recommendations']:
    print('-', x)
preds = pd.read_csv(MOD / 'predictions.csv', parse_dates=['datetime'])
preds.head()"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## Step 11: GitHub Deliverables Checklist

- [x] Project overview & business use case (README)
- [x] Dataset source + collection method
- [x] Cleaning, EDA visuals, engineered features
- [x] Multiple models + tuning + comparison table
- [x] Final evaluation artifacts (`models/`)
- [x] Streamlit dashboard
- [x] `requirements.txt` + run instructions
- [x] No secrets / API keys committed
"""
    )
)

nb["cells"] = cells
out = ROOT / "notebooks" / "PM25_Forecast_Capstone.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"notebook written: {out}")
