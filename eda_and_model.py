"""
End-to-end PM2.5 next-hour forecast pipeline (light scikit-learn models only).

Covers capstone steps:
  clean/prepare -> EDA -> feature engineering -> time-based split/CV
  -> multi-model training -> hyperparameter tuning -> evaluation
  -> artifacts for dashboard / notebook / GitHub deliverables

No XGBoost, LightGBM, CatBoost, or heavy gradient-boosting ensembles.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
VIS_DIR = ROOT / "visuals"
MODEL_DIR = ROOT / "models"
for d in (DATA_DIR, VIS_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def risk_category(v: float) -> str:
    if v <= 12:
        return "Good"
    if v <= 35.4:
        return "Moderate"
    if v <= 55.4:
        return "Unhealthy(SG)"
    return "Unhealthy"


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "MSE": round(float(mean_squared_error(y_true, y_pred)), 3),
        "RMSE": round(rmse(y_true, y_pred), 3),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


# ---------- STEP 3: CLEAN AND PREPARE ----------
raw_path = DATA_DIR / "pm25_raw.csv"
df = pd.read_csv(raw_path, parse_dates=["datetime"])
before = len(df)
missing_before = int(df["pm25"].isna().sum())

df = df.drop_duplicates()
after_dedup = len(df)

df["station_id"] = df["station_id"].astype(str)
df["datetime"] = pd.to_datetime(df["datetime"])
for col in ["pm25", "temperature_c", "humidity_pct", "wind_speed_mps"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.sort_values(["station_id", "datetime"]).reset_index(drop=True)
df["pm25"] = df.groupby("station_id")["pm25"].transform(
    lambda x: x.fillna(x.rolling(6, min_periods=1).mean())
)
df["pm25"] = df["pm25"].fillna(df["pm25"].mean())
for col in ["temperature_c", "humidity_pct", "wind_speed_mps"]:
    df[col] = df.groupby("station_id")[col].transform(lambda x: x.fillna(x.median()))
    df[col] = df[col].fillna(df[col].median())


def cap_outliers(s: pd.Series) -> pd.Series:
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    return s.clip(lo, hi)


for col in ["pm25", "temperature_c", "humidity_pct", "wind_speed_mps"]:
    df[col] = df.groupby("station_id")[col].transform(cap_outliers)

cleaning_log = {
    "raw_rows": before,
    "after_dedup": after_dedup,
    "duplicates_removed": before - after_dedup,
    "missing_pm25_before": missing_before,
    "final_rows_after_clean": len(df),
    "note": "Class imbalance N/A (regression). Scaling fit on train only.",
}

# ---------- STEP 5: FEATURE ENGINEERING ----------
df["hour"] = df["datetime"].dt.hour
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["dayofweek"] = df["datetime"].dt.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
df["season"] = df["month"].map(lambda m: (m % 12) // 3)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

df["temp_bin"] = pd.cut(
    df["temperature_c"],
    bins=[-50, 5, 15, 25, 50],
    labels=["cold", "cool", "mild", "warm"],
).astype(str)
df["wind_bin"] = pd.cut(
    df["wind_speed_mps"],
    bins=[-0.1, 2, 4, 6, 50],
    labels=["calm", "light", "moderate", "strong"],
).astype(str)

df["temp_x_humidity"] = df["temperature_c"] * df["humidity_pct"]
df["pm_proxy_weather"] = df["humidity_pct"] / (df["wind_speed_mps"] + 0.1)

df = df.sort_values(["station_id", "datetime"])
g = df.groupby("station_id")["pm25"]
df["pm25_lag1"] = g.shift(1)
df["pm25_lag3"] = g.shift(3)
df["pm25_lag24"] = g.shift(24)
df["pm25_roll6"] = g.transform(lambda x: x.rolling(6).mean())
df["pm25_roll24"] = g.transform(lambda x: x.rolling(24).mean())
df["pm25_roll6_std"] = g.transform(lambda x: x.rolling(6).std())
df["target_next_hour_pm25"] = g.shift(-1)

feature_cols_num = [
    "hour",
    "month",
    "day",
    "dayofweek",
    "is_weekend",
    "season",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "pm25_lag1",
    "pm25_lag3",
    "pm25_lag24",
    "pm25_roll6",
    "pm25_roll24",
    "pm25_roll6_std",
    "temperature_c",
    "humidity_pct",
    "wind_speed_mps",
    "temp_x_humidity",
    "pm_proxy_weather",
]
feature_cols_cat = ["station_id", "temp_bin", "wind_bin"]

df_model = df.dropna(subset=feature_cols_num + ["target_next_hour_pm25"]).copy()
df_model["pollution_risk"] = df_model["pm25"].apply(risk_category)
df_model.to_csv(DATA_DIR / "pm25_cleaned_features.csv", index=False)
cleaning_log["model_rows_after_feature_dropna"] = int(len(df_model))

# ---------- STEP 4: EDA VISUALS ----------
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(10, 4))
for st in df["station_id"].unique():
    sub = df[df["station_id"] == st].set_index("datetime")["pm25"].resample("D").mean()
    ax.plot(sub.index, sub.values, label=st, linewidth=0.9)
ax.set_title("Daily Average PM2.5 by Station")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.legend()
plt.tight_layout()
plt.savefig(VIS_DIR / "daily_pm25_trend.png", dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(8, 4))
df.groupby("hour")["pm25"].mean().plot(kind="bar", ax=ax, color="#3B7DD8")
ax.set_title("Average PM2.5 by Hour of Day")
ax.set_ylabel("PM2.5 (µg/m³)")
plt.tight_layout()
plt.savefig(VIS_DIR / "hourly_pattern.png", dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
corr = df[["pm25", "temperature_c", "humidity_pct", "wind_speed_mps"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax, fmt=".2f")
ax.set_title("Correlation Matrix")
plt.tight_layout()
plt.savefig(VIS_DIR / "correlation_matrix.png", dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
df["pm25"].plot(kind="hist", bins=40, ax=ax, color="#5CA85C", edgecolor="white")
ax.set_title("Distribution of PM2.5 Readings")
ax.set_xlabel("PM2.5 (µg/m³)")
plt.tight_layout()
plt.savefig(VIS_DIR / "pm25_distribution.png", dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
df.groupby("month")["pm25"].mean().plot(marker="o", ax=ax, color="#8B5CF6")
ax.set_title("Average PM2.5 by Month")
ax.set_xlabel("Month")
ax.set_ylabel("PM2.5 (µg/m³)")
plt.tight_layout()
plt.savefig(VIS_DIR / "monthly_pm25_trend.png", dpi=120)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, col, color in zip(
    axes,
    ["temperature_c", "humidity_pct", "wind_speed_mps"],
    ["#EF4444", "#3B82F6", "#10B981"],
):
    sample = df.sample(min(3000, len(df)), random_state=RANDOM_STATE)
    ax.scatter(sample[col], sample["pm25"], alpha=0.15, s=8, color=color)
    ax.set_xlabel(col)
    ax.set_ylabel("PM2.5")
    ax.set_title(f"PM2.5 vs {col}")
plt.tight_layout()
plt.savefig(VIS_DIR / "pm25_vs_weather.png", dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(x="station_id", y="pm25", data=df, ax=ax, color="#F59E0B")
ax.set_title("PM2.5 Outlier View by Station")
plt.tight_layout()
plt.savefig(VIS_DIR / "pm25_boxplot_outliers.png", dpi=120)
plt.close()

# ---------- STEP 6: TIME-BASED SPLIT ----------
df_model = df_model.sort_values("datetime").reset_index(drop=True)
split_idx = int(len(df_model) * 0.8)
train_df = df_model.iloc[:split_idx].copy()
test_df = df_model.iloc[split_idx:].copy()

X_train = train_df[feature_cols_num + feature_cols_cat]
y_train = train_df["target_next_hour_pm25"]
X_test = test_df[feature_cols_num + feature_cols_cat]
y_test = test_df["target_next_hour_pm25"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), feature_cols_num),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            feature_cols_cat,
        ),
    ]
)

X_train_trees = pd.get_dummies(X_train, columns=feature_cols_cat, drop_first=False)
X_test_trees = pd.get_dummies(X_test, columns=feature_cols_cat, drop_first=False)
X_test_trees = X_test_trees.reindex(columns=X_train_trees.columns, fill_value=0)

tscv = TimeSeriesSplit(n_splits=3)

# ---------- STEP 7: LIGHT MODELS ONLY ----------
base_models = {
    "Linear Regression": {
        "kind": "pipeline",
        "estimator": Pipeline([("prep", preprocess), ("model", LinearRegression())]),
        "X_train": X_train,
        "X_test": X_test,
    },
    "Ridge Regression": {
        "kind": "pipeline",
        "estimator": Pipeline(
            [("prep", preprocess), ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE))]
        ),
        "X_train": X_train,
        "X_test": X_test,
    },
    "ElasticNet": {
        "kind": "pipeline",
        "estimator": Pipeline(
            [
                ("prep", preprocess),
                (
                    "model",
                    ElasticNet(
                        alpha=0.01, l1_ratio=0.5, max_iter=3000, random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "X_train": X_train,
        "X_test": X_test,
    },
    "Decision Tree": {
        "kind": "tree",
        "estimator": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
        "X_train": X_train_trees,
        "X_test": X_test_trees,
    },
    "Random Forest": {
        "kind": "tree",
        "estimator": RandomForestRegressor(
            n_estimators=40, max_depth=8, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "X_train": X_train_trees,
        "X_test": X_test_trees,
    },
    "K-Nearest Neighbors": {
        "kind": "pipeline",
        "estimator": Pipeline(
            [("prep", preprocess), ("model", KNeighborsRegressor(n_neighbors=7))]
        ),
        "X_train": X_train,
        "X_test": X_test,
    },
    "Linear SVR": {
        "kind": "pipeline",
        "estimator": Pipeline(
            [
                ("prep", preprocess),
                ("model", LinearSVR(C=1.0, max_iter=2000, random_state=RANDOM_STATE)),
            ]
        ),
        "X_train": X_train,
        "X_test": X_test,
    },
    "MLP Neural Net": {
        "kind": "pipeline",
        "estimator": Pipeline(
            [
                ("prep", preprocess),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(32,),
                        max_iter=120,
                        random_state=RANDOM_STATE,
                        early_stopping=True,
                    ),
                ),
            ]
        ),
        "X_train": X_train,
        "X_test": X_test,
    },
}

results = []
preds_dict = {}
fitted_models = {}

for name, cfg in base_models.items():
    model = cfg["estimator"]
    Xt, Xte = cfg["X_train"], cfg["X_test"]

    t0 = time.perf_counter()
    model.fit(Xt, y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred_test = model.predict(Xte)
    pred_time = time.perf_counter() - t1
    pred_train = model.predict(Xt)

    test_metrics = evaluate(y_test, pred_test)
    train_metrics = evaluate(y_train, pred_train)

    cv_rmses = []
    if cfg["kind"] == "tree":
        Xcv = Xt
        for tr_idx, va_idx in tscv.split(Xcv):
            m = cfg["estimator"].__class__(**cfg["estimator"].get_params())
            m.fit(Xcv.iloc[tr_idx], y_train.iloc[tr_idx])
            p = m.predict(Xcv.iloc[va_idx])
            cv_rmses.append(rmse(y_train.iloc[va_idx], p))

    row = {
        "Model": name,
        **test_metrics,
        "Train_RMSE": train_metrics["RMSE"],
        "Train_R2": train_metrics["R2"],
        "Overfit_Gap_RMSE": round(train_metrics["RMSE"] - test_metrics["RMSE"], 3),
        "CV_RMSE_mean": round(float(np.mean(cv_rmses)), 3) if cv_rmses else None,
        "Train_time_sec": round(train_time, 3),
        "Predict_time_sec": round(pred_time, 4),
    }
    results.append(row)
    preds_dict[name] = pred_test
    fitted_models[name] = model
    print(f"Trained {name}: RMSE={row['RMSE']} R2={row['R2']}")

results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

# ---------- STEP 8: LIGHT TUNING (Random Forest + Decision Tree only) ----------
tune_specs = {
    "Random Forest": {
        "estimator": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "params": {
            "n_estimators": [30, 50, 70],
            "max_depth": [6, 8, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.8],
        },
    },
    "Decision Tree": {
        "estimator": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "params": {
            "max_depth": [4, 6, 8, 10],
            "min_samples_leaf": [1, 2, 5],
            "min_samples_split": [2, 5, 10],
        },
    },
}

tuned_rows = []
for name, spec in tune_specs.items():
    search = RandomizedSearchCV(
        estimator=spec["estimator"],
        param_distributions=spec["params"],
        n_iter=6,
        scoring="neg_root_mean_squared_error",
        cv=TimeSeriesSplit(n_splits=3),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    t0 = time.perf_counter()
    search.fit(X_train_trees, y_train)
    tune_time = time.perf_counter() - t0
    pred = search.predict(X_test_trees)
    metrics = evaluate(y_test, pred)
    train_metrics = evaluate(y_train, search.predict(X_train_trees))
    tuned_name = f"{name} (Tuned)"
    row = {
        "Model": tuned_name,
        **metrics,
        "Train_RMSE": train_metrics["RMSE"],
        "Train_R2": train_metrics["R2"],
        "Overfit_Gap_RMSE": round(train_metrics["RMSE"] - metrics["RMSE"], 3),
        "CV_RMSE_mean": round(float(-search.best_score_), 3),
        "Train_time_sec": round(tune_time, 3),
        "Predict_time_sec": None,
        "Best_Params": json.dumps(search.best_params_),
    }
    tuned_rows.append(row)
    preds_dict[tuned_name] = pred
    fitted_models[tuned_name] = search.best_estimator_
    print(f"Tuned {name}: RMSE={row['RMSE']} params={search.best_params_}")

if tuned_rows:
    results_df = (
        pd.concat([results_df, pd.DataFrame(tuned_rows)], ignore_index=True)
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

best_model_name = results_df.iloc[0]["Model"]
best_pred = preds_dict[best_model_name]
best_model = fitted_models[best_model_name]

results_df.to_csv(MODEL_DIR / "model_comparison.csv", index=False)

pred_out = test_df[["datetime", "station_id", "pm25", "target_next_hour_pm25"]].copy()
pred_out = pred_out.rename(columns={"target_next_hour_pm25": "actual_next_hour_pm25"})
pred_out["predicted_next_hour_pm25"] = best_pred
pred_out["abs_error"] = (
    pred_out["actual_next_hour_pm25"] - pred_out["predicted_next_hour_pm25"]
).abs()
pred_out["predicted_risk"] = pred_out["predicted_next_hour_pm25"].apply(risk_category)
pred_out["actual_risk"] = pred_out["actual_next_hour_pm25"].apply(risk_category)
pred_out.to_csv(MODEL_DIR / "predictions.csv", index=False)

tree_names = {
    "Decision Tree",
    "Random Forest",
    "Decision Tree (Tuned)",
    "Random Forest (Tuned)",
}
uses_tree = best_model_name in tree_names
bundle = {
    "model": best_model,
    "model_name": best_model_name,
    "feature_cols_num": feature_cols_num,
    "feature_cols_cat": feature_cols_cat,
    "tree_feature_columns": list(X_train_trees.columns) if uses_tree else None,
    "uses_tree_matrix": uses_tree,
}
joblib.dump(bundle, MODEL_DIR / "best_model.pkl")

# ---------- STEP 9: PLOTS ----------
fig, ax = plt.subplots(figsize=(10, 4.5))
plot_df = results_df.sort_values("RMSE")
colors = ["#059669" if m == best_model_name else "#D8823B" for m in plot_df["Model"]]
ax.barh(plot_df["Model"], plot_df["RMSE"], color=colors)
ax.set_xlabel("RMSE (lower is better)")
ax.set_title("Model Comparison — Test RMSE")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(VIS_DIR / "model_comparison_rmse.png", dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y_test.values[:250], label="Actual", linewidth=1.2)
ax.plot(best_pred[:250], label=f"Predicted ({best_model_name})", linewidth=1.1, alpha=0.85)
ax.set_title("Actual vs Predicted Next-Hour PM2.5 (first 250 test points)")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.legend()
plt.tight_layout()
plt.savefig(VIS_DIR / "actual_vs_predicted.png", dpi=120)
plt.close()

residuals = y_test.values - best_pred
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(best_pred, residuals, alpha=0.25, s=10, color="#6366F1")
axes[0].axhline(0, color="black", linewidth=1)
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Residual (Actual - Predicted)")
axes[0].set_title("Residual Plot")
axes[1].hist(residuals, bins=40, color="#F97316", edgecolor="white")
axes[1].set_title("Error Distribution")
axes[1].set_xlabel("Residual")
plt.tight_layout()
plt.savefig(VIS_DIR / "error_analysis.png", dpi=120)
plt.close()

importances = None
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(
        best_model.feature_importances_, index=list(X_train_trees.columns)
    ).sort_values(ascending=False)
elif "Random Forest" in fitted_models and hasattr(
    fitted_models["Random Forest"], "feature_importances_"
):
    importances = pd.Series(
        fitted_models["Random Forest"].feature_importances_,
        index=list(X_train_trees.columns),
    ).sort_values(ascending=False)

if importances is not None:
    fig, ax = plt.subplots(figsize=(8, 5))
    importances.head(15).plot(kind="barh", ax=ax, color="#3B9C8C")
    ax.invert_yaxis()
    ax.set_title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "feature_importance.png", dpi=120)
    plt.close()
    importances.head(20).to_csv(MODEL_DIR / "feature_importance.csv", header=["importance"])

best_row = results_df.iloc[0].to_dict()
summary = {
    "cleaning_log": cleaning_log,
    "split": {
        "strategy": "time-based 80/20 (no shuffle)",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "cv": "TimeSeriesSplit n_splits=3 (tree models + tuning)",
    },
    "features": {
        "numeric": feature_cols_num,
        "categorical": feature_cols_cat,
        "target": "target_next_hour_pm25",
        "leakage_guard": "Lags/rolling use past values only; target is next hour shift(-1)",
    },
    "models_trained": results_df["Model"].tolist(),
    "best_model": best_model_name,
    "best_metrics": {
        k: best_row[k]
        for k in ["MAE", "MSE", "RMSE", "R2", "Train_RMSE", "Overfit_Gap_RMSE"]
        if k in best_row
    },
    "results": results_df.drop(
        columns=[c for c in ["Best_Params"] if c in results_df.columns]
    ).to_dict(orient="records"),
    "tuning": {
        "method": "RandomizedSearchCV + TimeSeriesSplit",
        "candidates": list(tune_specs.keys()),
        "note": "Light models only — no XGBoost/LightGBM/CatBoost/heavy GBM",
    },
    "business_interpretation": (
        "Next-hour PM2.5 forecasts support health advisories and station-level risk flags. "
        "Lagged PM2.5 and short rolling means dominate predictive signal; weather adds secondary context."
    ),
    "limitations": [
        "Included dataset is synthetic/educational (EPA AQS schema style), not live AQS pulls.",
        "One-hour horizon only; multi-horizon forecasting not included.",
        "No live weather-forecast exogenous inputs.",
        "Station coverage limited to two demo stations.",
        "Light sklearn models only (no heavy gradient boosting libraries).",
    ],
    "recommendations": [
        "Replace sample data with real EPA AQS hourly downloads/API for production.",
        "Add more stations and multi-year history.",
        "Use forecast weather features from a weather API.",
        "Extend to 6h/24h horizons and calibrated risk alerts.",
        "Retrain periodically as new observations arrive.",
    ],
}

with open(MODEL_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, default=str)

print(
    json.dumps(
        {"best_model": best_model_name, "best_metrics": summary["best_metrics"]},
        indent=2,
    )
)
print("Artifacts written to data/, visuals/, models/")
