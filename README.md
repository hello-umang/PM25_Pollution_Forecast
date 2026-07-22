# US PM2.5 Pollution Forecast — ML Capstone Project

## 1. Project Overview
This project builds an end-to-end machine learning pipeline to forecast **next-hour PM2.5 pollution concentration** at US air quality monitoring stations, using time-series regression. It includes data cleaning, EDA, feature engineering, model training/comparison, and an interactive Streamlit dashboard.

## 2. Business Use Case
- **Problem:** Predict the next hour (or next day) PM2.5 concentration for an air monitoring station.
- **Target users:** Environmental agencies, public health officials, city/regional dashboards, and the general public checking local air quality.
- **Expected output:** PM2.5 forecast, pollution-risk category (Good / Moderate / Unhealthy for Sensitive Groups / Unhealthy), station comparison, and actual-vs-predicted graph.
- **Practical use case:** Early-warning alerts for high-pollution periods, supporting public health advisories (e.g. advising sensitive groups to limit outdoor activity), and city-level air quality dashboards.
- **Scope:** One US state (California, sample), 2 stations, 1 year of hourly data (~17,500 rows).

## 3. Dataset Source
- **Official source:** EPA Air Quality System (AQS) Data, US Environmental Protection Agency — https://www.epa.gov/aqs/obtaining-aqs-data
  - Hourly and daily data files, and the **AQS API**, are available directly from the EPA.
  - To pull real data: register for an AQS API key at https://aqs.epa.gov/data/api and query hourly PM2.5 (`param=88101`) for your state/county/site of interest.
- **This repository ships with locally generated sample data** (`data/pm25_raw.csv`) that mirrors the AQS hourly schema (station_id, datetime, pm25, temperature_c, humidity_pct, wind_speed_mps) for 2 California station IDs across all of 2023 (~17,500 rows), so the project can be run immediately without API keys or network access. Swap this file for a real AQS export to go to production — the pipeline (`eda_and_model.py`) will work unchanged as long as column names match.

## 4. Data-Cleaning Steps
1. Removed duplicate rows.
2. Filled missing PM2.5 values using a 6-hour rolling mean per station (fallback: overall mean).
3. Capped outliers at the 1st/99th percentile per station (winsorization).
4. Corrected data types (datetime parsing).
5. Removed leakage-prone raw index columns; final feature set built only from historical (lagged/rolling) values.

## 5. EDA & Visualisations
Located in `visuals/`:
- `daily_pm25_trend.png` — Daily average PM2.5 per station across the year (seasonality visible).
- `hourly_pattern.png` — Average PM2.5 by hour of day (rush-hour peaks).
- `correlation_matrix.png` — Correlation between PM2.5, temperature, humidity, wind speed.
- `pm25_distribution.png` — Histogram of PM2.5 readings.

**Key insights:**
- Winter months show higher average PM2.5 than summer (seasonal pattern).
- Morning/evening rush hours show elevated PM2.5 (diurnal/traffic pattern).
- Wind speed correlates negatively with PM2.5 (dispersion effect); temperature/humidity show mild relationships.

## 6. Feature Engineering
- Time features: hour, month, day-of-week, weekend flag, season.
- Lag features: `pm25_lag1` (previous hour), `pm25_lag24` (same hour previous day).
- Rolling features: 6-hour and 24-hour rolling averages.
- Target: `target_next_hour_pm25` (PM2.5 shifted **forward** by 1 hour — ensures no leakage since it uses only past/current data to predict the future).
- Derived label: `pollution_risk` category (Good / Moderate / Unhealthy(SG) / Unhealthy) based on standard PM2.5 breakpoints.

## 7. Models Tested
Light-weight models only were used (no XGBoost/LightGBM/CatBoost), per project scope, to minimize compute:
- Linear Regression
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor (small: 40 trees, depth 8)
- K-Nearest Neighbors Regressor

Validation strategy: **time-based 80:20 split** (no shuffling), appropriate for time-series forecasting. Preprocessing (scaling) fit only on training data.

## 8. Evaluation Results
See `models/model_comparison.csv` and `visuals/model_comparison_rmse.png`.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest (small) | 2.125 | 2.660 | 0.594 |
| Decision Tree | 2.382 | 2.989 | 0.487 |
| Linear Regression | 2.394 | 3.005 | 0.482 |
| Ridge Regression | 2.394 | 3.005 | 0.482 |
| K-Nearest Neighbors | 2.670 | 3.344 | 0.358 |

**Best model: Random Forest (small)** — lowest RMSE/MAE, highest R².

## 9. Final Comparison & Business Interpretation
The Random Forest model best captures the combined seasonal + diurnal + short-term autocorrelation structure of PM2.5, using lag/rolling features as the strongest predictors (see `visuals/feature_importance.png`). This is sufficient to power a 1-hour-ahead pollution-risk alert.

**Model limitations:**
- Based on 1 year, 2 stations (California, sample) — limited geographic/temporal generalization.
- Uses historical (not forecast) weather; a production system should pull forecast weather data.
- No boosting models tested (XGBoost/LightGBM) — left as a recommended next step once more compute is available.

**Recommendations for improvement:**
1. Connect to the real EPA AQS API for live/historical data at scale.
2. Add more stations/years for broader generalization.
3. Incorporate weather **forecast** (not just historical) features.
4. Test gradient boosting models (XGBoost/LightGBM) for potential accuracy gains.
5. Retrain on a schedule (e.g., weekly) as new AQS data becomes available.

## 10. Dashboard / Demo
A Streamlit dashboard (`dashboard/app.py`) shows:
- Best-performing model and metrics
- Current pollution-risk status per station
- PM2.5 trend and actual-vs-predicted chart
- Model comparison chart
- Feature importance
- Station comparison

## 11. Project Structure
```
pm25_project/
├── README.md
├── requirements.txt
├── generate_data.py          # generates sample AQS-style dataset
├── eda_and_model.py           # cleaning, EDA, feature engineering, training, evaluation
├── build_notebook.py          # builds the Jupyter notebook programmatically
├── data/
│   ├── pm25_raw.csv
│   └── pm25_cleaned_features.csv
├── notebooks/
│   └── PM25_Forecast_Capstone.ipynb
├── visuals/
│   ├── daily_pm25_trend.png
│   ├── hourly_pattern.png
│   ├── correlation_matrix.png
│   ├── pm25_distribution.png
│   ├── model_comparison_rmse.png
│   ├── actual_vs_predicted.png
│   └── feature_importance.png
├── models/
│   ├── model_comparison.csv
│   └── summary.json
└── dashboard/
    └── app.py                  # Streamlit dashboard
```

## 12. Instructions to Run

### Setup
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Regenerate data (optional — data already included)
```bash
python generate_data.py
```

### Run the full pipeline (cleaning, EDA, training, evaluation)
```bash
python eda_and_model.py
```

### Open the notebook
```bash
jupyter notebook notebooks/PM25_Forecast_Capstone.ipynb
```

### Run the dashboard
```bash
streamlit run dashboard/app.py
```

## 13. Publishing to GitHub
1. Create a new **public** repository on GitHub.
2. `git init`, `git add .`, `git commit -m "Initial commit: PM2.5 forecast capstone"`.
3. `git remote add origin <your-repo-url>` and `git push -u origin main`.
4. Confirm no passwords, API keys, or confidential data are included (this repo contains none — sample data only).
5. Add the repository link and dashboard/demo link to your submission (Google Classroom).

## 14. Submission Checklist
- [x] Public GitHub repository link
- [x] Jupyter Notebook / Python project (`notebooks/`, `eda_and_model.py`)
- [x] EDA and visualisations (`visuals/`)
- [x] Model comparison table (`models/model_comparison.csv`)
- [x] Final model evaluation (`models/summary.json`, README section 8-9)
- [x] Dashboard / working demo (`dashboard/app.py`)
- [x] README with findings, limitations, and business recommendations (this file)

---
*No passwords, API keys, or confidential/personal data are included in this repository.*
