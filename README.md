# PM2.5 Pollution Forecast Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)
![License](https://img.shields.io/badge/License-Educational-green.svg)

End-to-end **regression / time-series forecasting** capstone: predict **next-hour PM2.5** from EPA AQS-style hourly air-quality + weather features, with EDA, multi-model training, hyperparameter tuning, evaluation, and a Streamlit dashboard.

> **Light scikit-learn models only** — no XGBoost, LightGBM, CatBoost, or heavy gradient-boosting libraries.

---

## 1. Problem Definition (Step 1)

| Item | Detail |
|------|--------|
| **Business problem** | Agencies need short-horizon PM2.5 forecasts to issue health advisories and monitor stations |
| **ML task** | Regression / forecasting |
| **Target users** | Environmental agencies, pollution control boards, smart-city teams, researchers, public-health planners |
| **Expected output** | Next-hour PM2.5 (µg/m³), risk category, model comparison, dashboard |
| **Practical use case** | Station-level early warning and operational air-quality monitoring |

**Target variable:** `target_next_hour_pm25` (next hour’s PM2.5 concentration).

---

## 2. Data Collection (Step 2)

### Official source pattern
- **United States EPA — Air Quality System (AQS)**  
  https://www.epa.gov/aqs/obtaining-aqs-data  
- Parameter code **88101** (PM2.5 Local Conditions)

### Dataset included (educational / offline)
| File | Description |
|------|-------------|
| `data/pm25_raw.csv` | Synthetic hourly dataset (≥10k rows) matching AQS-style schema |
| `data/pm25_cleaned_features.csv` | Cleaned + engineered modeling table |

**Collection method:** `generate_data.py` creates a reproducible synthetic year of hourly data for **2 stations** so the project runs without API keys. Swap in real AQS downloads/API for production.

| Column | Description |
|--------|-------------|
| `station_id` | Monitoring station identifier |
| `datetime` | Observation timestamp |
| `pm25` | PM2.5 concentration (µg/m³) |
| `temperature_c` | Temperature (°C) |
| `humidity_pct` | Relative humidity (%) |
| `wind_speed_mps` | Wind speed (m/s) |

---

## 3. Project Structure

```text
PM25_Pollution_Forecast/
├── data/
│   ├── pm25_raw.csv
│   └── pm25_cleaned_features.csv
├── dashboard/
│   └── app.py
├── models/
│   ├── best_model.pkl
│   ├── model_comparison.csv
│   ├── predictions.csv
│   ├── feature_importance.csv
│   └── summary.json
├── notebooks/
│   └── PM25_Forecast_Capstone.ipynb
├── visuals/                      # EDA + evaluation charts
├── generate_data.py
├── eda_and_model.py
├── build_notebook.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 4. Machine Learning Workflow (Steps 3–9)

### Step 3 — Clean & prepare
- Remove duplicates  
- Handle missing PM2.5 (station rolling mean → global mean)  
- Correct dtypes / sort by station + time  
- Percentile outlier capping (1st–99th) per station  
- Encode categoricals + scale numerics **inside train-only pipelines**  
- Class imbalance: **N/A** (regression)

### Step 4 — EDA
Charts in `visuals/`:
- Daily / monthly trends, hourly pattern  
- PM2.5 distribution & station boxplots  
- Correlation matrix  
- PM2.5 vs weather scatter panel  

Each chart’s key insight is documented in the notebook.

### Step 5 — Feature engineering
- Calendar: hour, day, month, weekday, weekend, season  
- Cyclical encodings: `hour_sin/cos`, `month_sin/cos`  
- Lags / rolling: lag-1/3/24, roll-6/24 mean, roll-6 std  
- Bins: temperature & wind bins  
- Interactions: temp×humidity, humidity/(wind+ε)  
- Target: next-hour PM2.5 (`shift(-1)`)  
- **Leakage guard:** lags/rolling use past values only

### Step 6 — Split & validate
- Chronological **80:20** train/test split (no shuffle)  
- **TimeSeriesSplit** (3 folds) for tree-model CV and tuning  
- Scaler / one-hot fit on training folds only

### Step 7 — Models trained (light sklearn only)
1. Linear Regression  
2. Ridge Regression  
3. ElasticNet  
4. Decision Tree  
5. Random Forest (small)  
6. K-Nearest Neighbors  
7. Linear SVR  
8. MLP Neural Net (small)  

Tuned variants of Decision Tree and Random Forest are also reported.

### Step 8 — Hyperparameter tuning
- **RandomizedSearchCV** + **TimeSeriesSplit** on Decision Tree & Random Forest  
- Search space examples: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`, `min_samples_split`

### Step 9 — Evaluation
| Metric | Purpose |
|--------|---------|
| MAE / MSE / RMSE | Error magnitude |
| R² | Explained variance |
| Train vs test RMSE gap | Over/underfitting check |
| CV RMSE (trees) | Time-series stability |
| Train / predict time | Operational cost |

Artifacts:
- `models/model_comparison.csv`  
- `models/predictions.csv`  
- `models/best_model.pkl`  
- `models/summary.json`  
- `visuals/model_comparison_rmse.png`, `actual_vs_predicted.png`, `feature_importance.png`, `error_analysis.png`

---

## 5. Dashboard (Step 10)

```bash
streamlit run dashboard/app.py
```

Shows best-model KPIs, station filters, trends, comparison table, feature importance, error analysis, limitations, and recommendations.

---

## 6. Setup & Run

```bash
# clone
git clone https://github.com/hello-umang/PM25_Pollution_Forecast.git
cd PM25_Pollution_Forecast

# install
pip install -r requirements.txt

# optional: regenerate sample data
python generate_data.py

# full clean → EDA → train → tune → evaluate → save artifacts
python eda_and_model.py

# rebuild notebook stubs that display saved artifacts
python build_notebook.py

# dashboard
streamlit run dashboard/app.py
```

### Google Colab (class demo)

**Easiest:** open the one-cell reset notebook from GitHub:

https://colab.research.google.com/github/hello-umang/PM25_Pollution_Forecast/blob/main/notebooks/Colab_Nuclear_Reset.ipynb

Or in Colab: **File → Open notebook → GitHub** → `hello-umang/PM25_Pollution_Forecast` → open `notebooks/Colab_Nuclear_Reset.ipynb` → **Runtime → Run all**.

That notebook:
1. Deletes any old `PM25_Pollution_Forecast` folder in Colab  
2. Fresh-clones this repo  
3. Installs dependencies  
4. Regenerates data + trains models  
5. Prints a full process log, comparison table, and charts  

Manual Colab cells (alternative):

```python
!git clone https://github.com/hello-umang/PM25_Pollution_Forecast.git
%cd PM25_Pollution_Forecast
!pip install -q numpy pandas matplotlib seaborn scikit-learn joblib scipy plotly
!python generate_data.py
!python eda_and_model.py
```

> Streamlit dashboard is meant for local run (`streamlit run dashboard/app.py`), not Colab.

---

## 7. Results & Business Interpretation

After `eda_and_model.py`, open:
- `models/model_comparison.csv` — full leaderboard  
- `models/summary.json` — best model, metrics, cleaning log, recommendations  

**Interpretation:** Next-hour forecasts support short-term health messaging and station monitoring. Lagged PM2.5 and short rolling means usually dominate; weather adds secondary context. Residual/error plots show harder cases on spike hours.

**Latest run (light models):** best = **Linear Regression** — RMSE **2.747**, MAE **2.193**, R² **0.598** (see `models/model_comparison.csv` for the full leaderboard).

---

## 8. Limitations

- Included data is **synthetic/educational** (AQS schema style), not a live AQS extract  
- Single **1-hour** horizon  
- Only **two** demo stations  
- No live weather-forecast exogenous inputs  
- Spike / anomaly hours remain harder to predict

## 9. Recommendations

1. Replace sample data with real EPA AQS hourly downloads/API  
2. Add more stations and multi-year history  
3. Add forecast weather features from a weather API  
4. Extend to 6h/24h horizons and calibrated risk alerts  
5. Retrain periodically as new observations arrive  

---

## 10. GitHub Publish Checklist (Step 11)

- [x] Project overview & business use case  
- [x] Dataset source & collection method  
- [x] Cleaning steps  
- [x] EDA visualisations  
- [x] Models tested + tuning  
- [x] Evaluation / comparison table  
- [x] Dashboard  
- [x] README + `requirements.txt` + run instructions  
- [x] No passwords, API keys, or personal data  

**Repository:** https://github.com/hello-umang/PM25_Pollution_Forecast  

---

## Final Deliverables Mapping

| Deliverable | Location |
|-------------|----------|
| Public GitHub repo | link above |
| Jupyter notebook / Python project | `notebooks/`, `*.py` |
| EDA & visualisations | `visuals/` |
| Model comparison table | `models/model_comparison.csv` |
| Final model evaluation | `models/summary.json`, `best_model.pkl`, `predictions.csv` |
| Dashboard | `dashboard/app.py` |
| README findings / limits / recommendations | this file |

---

## License

Educational / academic use.

## Acknowledgements

EPA AQS, scikit-learn, pandas, NumPy, Matplotlib, Seaborn, Streamlit.

## Author

**Umang** — Machine Learning Capstone Project
