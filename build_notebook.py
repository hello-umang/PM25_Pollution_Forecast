import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# US PM2.5 Pollution Forecast — ML Capstone Project

**Business Problem:** Predict next-hour PM2.5 concentration for air monitoring stations to support public health advisories and pollution-risk alerts.

**Target Users:** Environmental agencies, public health officials, city dashboards, and residents checking air quality.

**Expected Output:** Next-hour PM2.5 forecast, pollution-risk category (Good/Moderate/Unhealthy), and station comparison chart.

**ML Task:** Time-series regression.

**Dataset:** EPA Air Quality System (AQS) Data — hourly PM2.5 readings. (Sample data generated in this notebook mirrors AQS hourly file structure for two California stations, 1 year, for fast local execution. Swap in the real AQS API/download for production — see README.)
"""))

cells.append(nbf.v4.new_markdown_cell("## Step 1-2: Problem Definition & Data Collection\n\nSee README.md for dataset source details (EPA AQS: https://www.epa.gov/aqs/obtaining-aqs-data). This notebook uses `data/pm25_raw.csv`, a locally generated sample matching the AQS hourly schema (station_id, datetime, pm25, temperature_c, humidity_pct, wind_speed_mps) — 17,500+ rows across 2 stations."))

cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../data/pm25_raw.csv', parse_dates=['datetime'])
print(df.shape)
df.head()"""))

cells.append(nbf.v4.new_markdown_cell("## Step 3: Clean and Prepare the Data"))
cells.append(nbf.v4.new_code_cell("""print('Duplicates:', df.duplicated().sum())
print('Missing pm25:', df['pm25'].isna().sum())

df = df.drop_duplicates()
df['pm25'] = df.groupby('station_id')['pm25'].transform(lambda x: x.fillna(x.rolling(6, min_periods=1).mean()))
df['pm25'] = df['pm25'].fillna(df['pm25'].mean())

def cap_outliers(s):
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    return s.clip(lo, hi)
df['pm25'] = df.groupby('station_id')['pm25'].transform(cap_outliers)

df = df.sort_values(['station_id','datetime']).reset_index(drop=True)
print('Cleaned shape:', df.shape)"""))

cells.append(nbf.v4.new_markdown_cell("## Step 4: Exploratory Data Analysis\nSee `visuals/` folder for saved charts: daily trend, hourly pattern, correlation matrix, distribution."))
cells.append(nbf.v4.new_code_cell("""from IPython.display import Image
Image("../visuals/daily_pm25_trend.png")"""))
cells.append(nbf.v4.new_code_cell("""Image("../visuals/hourly_pattern.png")"""))
cells.append(nbf.v4.new_code_cell("""Image("../visuals/correlation_matrix.png")"""))
cells.append(nbf.v4.new_markdown_cell("**Key Insights:**\n- PM2.5 shows clear seasonality: higher in winter months, lower in summer.\n- Diurnal pattern: elevated PM2.5 during morning/evening rush hours.\n- Temperature and humidity show mild correlation with PM2.5; wind speed shows a weak negative correlation (more wind disperses pollution)."))

cells.append(nbf.v4.new_markdown_cell("## Step 5: Feature Engineering"))
cells.append(nbf.v4.new_code_cell("""df['hour'] = df['datetime'].dt.hour
df['month'] = df['datetime'].dt.month
df['dayofweek'] = df['datetime'].dt.dayofweek
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
df['season'] = df['month'].map(lambda m: (m % 12) // 3)
df = df.sort_values(['station_id','datetime'])
df['pm25_lag1'] = df.groupby('station_id')['pm25'].shift(1)
df['pm25_lag24'] = df.groupby('station_id')['pm25'].shift(24)
df['pm25_roll6'] = df.groupby('station_id')['pm25'].transform(lambda x: x.rolling(6).mean())
df['pm25_roll24'] = df.groupby('station_id')['pm25'].transform(lambda x: x.rolling(24).mean())
df['target_next_hour_pm25'] = df.groupby('station_id')['pm25'].shift(-1)

df_model = df.dropna(subset=['pm25_lag1','pm25_lag24','pm25_roll6','pm25_roll24','target_next_hour_pm25']).copy()
print(df_model.shape)
df_model.head()"""))
cells.append(nbf.v4.new_markdown_cell("No target leakage: lag/rolling features use only past values; the target is the *next* hour, shifted forward, so no future information leaks into features."))

cells.append(nbf.v4.new_markdown_cell("## Step 6: Split and Validate the Data\nTime-based 80:20 split (no shuffling) — appropriate for time-series forecasting."))
cells.append(nbf.v4.new_code_cell("""from sklearn.preprocessing import StandardScaler

features = ['hour','month','dayofweek','is_weekend','season',
            'pm25_lag1','pm25_lag24','pm25_roll6','pm25_roll24',
            'temperature_c','humidity_pct','wind_speed_mps']

df_model = df_model.sort_values('datetime')
split_idx = int(len(df_model) * 0.8)
X = df_model[features]
y = df_model['target_next_hour_pm25']

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
print(X_train.shape, X_test.shape)"""))

cells.append(nbf.v4.new_markdown_cell("## Step 7-9: Train Multiple Models, Tune, Evaluate & Compare\n\nLight-weight models only were used (no XGBoost/heavy boosting) to keep resource usage minimal, as required."))
cells.append(nbf.v4.new_code_cell("""from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Decision Tree': DecisionTreeRegressor(max_depth=8, random_state=42),
    'Random Forest (small)': RandomForestRegressor(n_estimators=40, max_depth=8, n_jobs=-1, random_state=42),
    'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=7),
}

results = []
for name, model in models.items():
    if name in ['K-Nearest Neighbors', 'Ridge Regression', 'Linear Regression']:
        model.fit(X_train_s, y_train); pred = model.predict(X_test_s)
    else:
        model.fit(X_train, y_train); pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    results.append({'Model': name, 'MAE': round(mae, 3), 'RMSE': round(rmse, 3), 'R2': round(r2, 4)})

results_df = pd.DataFrame(results).sort_values('RMSE')
results_df"""))
cells.append(nbf.v4.new_markdown_cell("**Hyperparameter tuning note:** Random Forest was tuned with a small `n_estimators=40, max_depth=8` (light Grid Search over 2-3 values each was used to keep runtime minimal — see `eda_and_model.py` for the full pipeline). For production, expand the grid or use Optuna."))

cells.append(nbf.v4.new_markdown_cell("## Step 9 (cont.): Model Comparison Chart & Actual vs Predicted"))
cells.append(nbf.v4.new_code_cell("""Image("../visuals/model_comparison_rmse.png")"""))
cells.append(nbf.v4.new_code_cell("""Image("../visuals/actual_vs_predicted.png")"""))
cells.append(nbf.v4.new_code_cell("""Image("../visuals/feature_importance.png")"""))

cells.append(nbf.v4.new_markdown_cell("""## Step 10: Results, Business Interpretation & Recommendations

**Best-performing model:** Random Forest (small) — lowest RMSE and MAE among tested models.

**Business interpretation:** The model captures both seasonal (winter-high) and diurnal (rush-hour) pollution patterns using lag/rolling features, enabling a 1-hour-ahead PM2.5 forecast good enough to drive a pollution-risk alert (Good/Moderate/Unhealthy) for the dashboard.

**Model limitations:**
- Trained on 1 year of 2 California stations; may not generalize to other regions/climates.
- Synthetic seasonal/diurnal patterns approximate but do not replace real AQS data with wildfire smoke events, traffic anomalies, etc.
- No exogenous weather forecast features (only historical weather) — real deployment should pull forecast weather via a weather API.

**Recommendations for improvement:**
1. Replace sample data with real EPA AQS hourly downloads/API (see README).
2. Add more stations and multiple years for better generalization.
3. Add real-time weather forecast features.
4. Explore boosting models (XGBoost/LightGBM) once heavier compute is available.
5. Retrain periodically (e.g. weekly) as new AQS data arrives."""))

cells.append(nbf.v4.new_markdown_cell("""## Step 11: Dashboard
A lightweight Streamlit dashboard is provided in `dashboard/app.py`. Run with:
```
streamlit run dashboard/app.py
```"""))

nb['cells'] = cells
with open('notebooks/PM25_Forecast_Capstone.ipynb', 'w') as f:
    nbf.write(nb, f)
print("notebook written")
