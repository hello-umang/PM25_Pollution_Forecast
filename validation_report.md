# Capstone Validation Report

Source PDF: Project_Steps.pdf
Zip: PM25_Pollution_Forecast.zip
**Score: 72/72 (100.0%)**

## Gaps

None — all automated checks passed.

## Step-by-step results

### Deliverable (7/7)
- ✅ Jupyter notebook or Python project — build_notebook.py, eda_and_model.py, generate_data.py, dashboard/app.py, notebooks/PM25_Forecast_Capstone.ipynb
- ✅ EDA and visualisations — 11 visuals
- ✅ Model comparison table
- ✅ Final model evaluation
- ✅ Dashboard or working demo
- ✅ README with findings/limitations/recommendations
- ✅ requirements.txt

### Step 1 (4/4)
- ✅ Business problem defined
- ✅ Target users defined
- ✅ Expected output / use case
- ✅ Task type (regression/forecasting)

### Step 2 (3/3)
- ✅ Dataset >= 10,000 rows — rows=17555 cols=['station_id', 'datetime', 'pm25', 'temperature_c', 'humidity_pct', 'wind_speed_mps']
- ✅ Dataset source mentioned
- ✅ Collection method mentioned

### Step 3 (8/8)
- ✅ Remove duplicates
- ✅ Handle missing values
- ✅ Correct data types
- ✅ Detect/treat outliers
- ✅ Encode categorical variables
- ✅ Scale/normalise numerical features
- ✅ Handle class imbalance (or N/A for regression)
- ✅ Remove leakage-prone columns / leakage guard

### Step 4 (8/8)
- ✅ EDA visual: distribution
- ✅ EDA visual: correlation
- ✅ EDA visual: trends
- ✅ EDA visual: patterns
- ✅ EDA visual: outliers
- ✅ EDA visual: relationships
- ✅ EDA visual: actual_vs_pred
- ✅ Key insight explained per important chart

### Step 5 (6/6)
- ✅ Derived / datetime features
- ✅ Lag/rolling features
- ✅ Binning
- ✅ Transformations / cyclical
- ✅ Interaction features
- ✅ Feature importance

### Step 6 (3/3)
- ✅ Train/test split present
- ✅ Time-series validation
- ✅ Preprocessing fit on train only

### Step 7 (3/3)
- ✅ Multiple models trained — count=10 models=['Linear Regression', 'Ridge Regression', 'Linear SVR', 'ElasticNet', 'Random Forest', 'Random Forest (Tuned)', 'Decision Tree (Tuned)', 'Decision Tree', 'MLP Neural Net', 'K-Nearest Neighbors']
- ✅ No heavy boosters (XGBoost/LGBM/CatBoost) trained — models=['Linear Regression', 'Ridge Regression', 'Linear SVR', 'ElasticNet', 'Random Forest', 'Random Forest (Tuned)', 'Decision Tree (Tuned)', 'Decision Tree', 'MLP Neural Net', 'K-Nearest Neighbors']
- ✅ No forbidden booster imports in training code

### Step 8 (2/2)
- ✅ Hyperparameter tuning present
- ✅ Tuned model(s) in comparison — ['Random Forest (Tuned)', 'Decision Tree (Tuned)']

### Step 9 (8/8)
- ✅ Metric column: MAE — ['Model', 'MAE', 'MSE', 'RMSE', 'R2', 'Train_RMSE', 'Train_R2', 'Overfit_Gap_RMSE', 'CV_RMSE_mean', 'Train_time_sec', 'Predict_time_sec', 'Best_Params']
- ✅ Metric column: RMSE — ['Model', 'MAE', 'MSE', 'RMSE', 'R2', 'Train_RMSE', 'Train_R2', 'Overfit_Gap_RMSE', 'CV_RMSE_mean', 'Train_time_sec', 'Predict_time_sec', 'Best_Params']
- ✅ Metric column: R2 — ['Model', 'MAE', 'MSE', 'RMSE', 'R2', 'Train_RMSE', 'Train_R2', 'Overfit_Gap_RMSE', 'CV_RMSE_mean', 'Train_time_sec', 'Predict_time_sec', 'Best_Params']
- ✅ Metric column: MSE — ['Model', 'MAE', 'MSE', 'RMSE', 'R2', 'Train_RMSE', 'Train_R2', 'Overfit_Gap_RMSE', 'CV_RMSE_mean', 'Train_time_sec', 'Predict_time_sec', 'Best_Params']
- ✅ Comparison table file
- ✅ Comparison chart
- ✅ Overfitting check
- ✅ Training/prediction time tracked

### Step 10 (8/8)
- ✅ Best model identified
- ✅ Actual vs predicted
- ✅ Feature importance
- ✅ Error analysis
- ✅ Business interpretation
- ✅ Model limitations
- ✅ Recommendations
- ✅ Dashboard (Streamlit/etc)

### Step 11 (4/4)
- ✅ Project overview in README
- ✅ Run instructions
- ✅ No secrets/API keys in repo files
- ✅ .gitignore present

### Results (3/3)
- ✅ Best model recorded — Linear Regression
- ✅ Best metrics recorded — {"MAE": 2.193, "MSE": 7.544, "RMSE": 2.747, "R2": 0.5977, "Train_RMSE": 2.579, "Overfit_Gap_RMSE": -0.168}
- ✅ Cleaning log present — {"raw_rows": 17555, "after_dedup": 17520, "duplicates_removed": 35, "missing_pm25_before": 175, "final_rows_after_clean": 17520, "note": "Class imbalance N/A (regression). Scaling fit on train only.", "model_rows_after_feature_dropna": 17470}

### Quality (5/5)
- ✅ requirements.txt is plain deps (not fenced)
- ✅ No xgboost dependency required
- ✅ best_model.pkl present
- ✅ predictions.csv present
- ✅ Cleaned feature table usable — rows=17470 n_cols=28

## Latest model results

- Best model: **Linear Regression**
- Metrics: `{"MAE": 2.193, "MSE": 7.544, "RMSE": 2.747, "R2": 0.5977, "Train_RMSE": 2.579, "Overfit_Gap_RMSE": -0.168}`
- Models trained: 10

### Comparison table

| Model | MAE | MSE | RMSE | R2 | Train_RMSE | Train_R2 | Overfit_Gap_RMSE | CV_RMSE_mean | Train_time_sec | Predict_time_sec | Best_Params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Linear Regression | 2.193 | 7.544 | 2.747 | 0.5977 | 2.579 | 0.7924 | -0.168 |  | 0.075 | 0.0093 |  |
| Ridge Regression | 2.193 | 7.543 | 2.747 | 0.5977 | 2.579 | 0.7924 | -0.168 |  | 0.035 | 0.0085 |  |
| Linear SVR | 2.203 | 7.605 | 2.758 | 0.5945 | 2.582 | 0.792 | -0.176 |  | 0.216 | 0.0099 |  |
| ElasticNet | 2.212 | 7.652 | 2.766 | 0.5919 | 2.584 | 0.7917 | -0.182 |  | 0.092 | 0.0086 |  |
| Random Forest | 2.227 | 7.732 | 2.781 | 0.5877 | 2.319 | 0.8321 | -0.462 | 3.27 | 2.84 | 0.0261 |  |
| Random Forest (Tuned) | 2.231 | 7.734 | 2.781 | 0.5876 | 2.505 | 0.8041 | -0.276 | 3.33 | 18.754 |  | {"n_estimators": 30, "min_samples_leaf": |
| Decision Tree (Tuned) | 2.28 | 8.087 | 2.844 | 0.5688 | 2.552 | 0.7967 | -0.292 | 3.348 | 1.383 |  | {"min_samples_split": 5, "min_samples_le |
| Decision Tree | 2.376 | 8.962 | 2.994 | 0.5221 | 2.411 | 0.8186 | -0.583 | 3.785 | 0.205 | 0.0037 |  |
| MLP Neural Net | 2.429 | 9.154 | 3.025 | 0.5119 | 2.508 | 0.8038 | -0.517 |  | 3.007 | 0.0097 |  |
| K-Nearest Neighbors | 2.784 | 12.151 | 3.486 | 0.352 | 2.442 | 0.814 | -1.044 |  | 0.024 | 0.3755 |  |

## PDF Final Deliverables Mapping

| PDF deliverable | Project location | Status |
|---|---|---|
| Public GitHub repository link | README | ✅ documented |
| Jupyter Notebook or Python project | notebooks/ + *.py | ✅ |
| EDA and visualisations | visuals/ (11 files) | ✅ |
| Model comparison table | models/model_comparison.csv | ✅ |
| Final model evaluation | models/summary.json, best_model.pkl, predictions.csv | ✅ |
| Dashboard or working demo | dashboard/app.py | ✅ |
| README findings/limitations/recommendations | README.md | ✅ |
