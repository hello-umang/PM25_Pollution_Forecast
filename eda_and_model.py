import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import warnings
warnings.filterwarnings("ignore")

# ---------- STEP 3: CLEAN DATA ----------
df = pd.read_csv("data/pm25_raw.csv", parse_dates=["datetime"])
before = len(df)
df = df.drop_duplicates()
after_dedup = len(df)
df["pm25"] = df.groupby("station_id")["pm25"].transform(lambda x: x.fillna(x.rolling(6, min_periods=1).mean()))
df["pm25"] = df["pm25"].fillna(df["pm25"].mean())
df = df.sort_values(["station_id","datetime"]).reset_index(drop=True)

# outlier treatment (cap at 1st/99th percentile per station)
def cap_outliers(s):
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    return s.clip(lo, hi)
df["pm25"] = df.groupby("station_id")["pm25"].transform(cap_outliers)

cleaning_log = {
    "raw_rows": before,
    "after_dedup": after_dedup,
    "duplicates_removed": before - after_dedup,
    "missing_pm25_before": int(pd.read_csv('data/pm25_raw.csv')['pm25'].isna().sum()),
    "final_rows": len(df)
}

# ---------- STEP 5: FEATURE ENGINEERING ----------
df["hour"] = df["datetime"].dt.hour
df["month"] = df["datetime"].dt.month
df["dayofweek"] = df["datetime"].dt.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
df["season"] = df["month"].map(lambda m: (m%12)//3)  # 0=winter..3=fall approx
df = df.sort_values(["station_id","datetime"])
df["pm25_lag1"] = df.groupby("station_id")["pm25"].shift(1)
df["pm25_lag24"] = df.groupby("station_id")["pm25"].shift(24)
df["pm25_roll6"] = df.groupby("station_id")["pm25"].transform(lambda x: x.rolling(6).mean())
df["pm25_roll24"] = df.groupby("station_id")["pm25"].transform(lambda x: x.rolling(24).mean())
df["target_next_hour_pm25"] = df.groupby("station_id")["pm25"].shift(-1)

df_model = df.dropna(subset=["pm25_lag1","pm25_lag24","pm25_roll6","pm25_roll24","target_next_hour_pm25"]).copy()

def risk_category(v):
    if v <= 12: return "Good"
    elif v <= 35.4: return "Moderate"
    elif v <= 55.4: return "Unhealthy(SG)"
    else: return "Unhealthy"
df_model["pollution_risk"] = df_model["pm25"].apply(risk_category)

df_model.to_csv("data/pm25_cleaned_features.csv", index=False)

# ---------- STEP 4: EDA VISUALS ----------
fig, ax = plt.subplots(figsize=(10,4))
for st in df["station_id"].unique():
    sub = df[df["station_id"]==st].set_index("datetime")["pm25"].resample("D").mean()
    ax.plot(sub.index, sub.values, label=st, linewidth=0.8)
ax.set_title("Daily Average PM2.5 by Station (2023)")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.legend()
plt.tight_layout()
plt.savefig("visuals/daily_pm25_trend.png", dpi=110)
plt.close()

fig, ax = plt.subplots(figsize=(8,4))
df.groupby("hour")["pm25"].mean().plot(kind="bar", ax=ax, color="#3B7DD8")
ax.set_title("Average PM2.5 by Hour of Day")
ax.set_ylabel("PM2.5 (µg/m³)")
plt.tight_layout()
plt.savefig("visuals/hourly_pattern.png", dpi=110)
plt.close()

fig, ax = plt.subplots(figsize=(6,5))
corr = df[["pm25","temperature_c","humidity_pct","wind_speed_mps"]].corr()
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("Correlation Matrix")
plt.colorbar(im)
plt.tight_layout()
plt.savefig("visuals/correlation_matrix.png", dpi=110)
plt.close()

fig, ax = plt.subplots(figsize=(6,4))
df["pm25"].plot(kind="hist", bins=40, ax=ax, color="#5CA85C")
ax.set_title("Distribution of PM2.5 Readings")
ax.set_xlabel("PM2.5 (µg/m³)")
plt.tight_layout()
plt.savefig("visuals/pm25_distribution.png", dpi=110)
plt.close()

# ---------- STEP 6: TRAIN/TEST SPLIT (time-based) ----------
df_model = df_model.sort_values("datetime")
split_idx = int(len(df_model)*0.8)
features = ["hour","month","dayofweek","is_weekend","season",
            "pm25_lag1","pm25_lag24","pm25_roll6","pm25_roll24",
            "temperature_c","humidity_pct","wind_speed_mps"]
X = df_model[features]
y = df_model["target_next_hour_pm25"]

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------- STEP 7-9: TRAIN LIGHT MODELS + EVALUATE ----------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest (small)": RandomForestRegressor(n_estimators=40, max_depth=8, n_jobs=-1, random_state=42),
    "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=7),
}

results = []
preds_dict = {}
for name, model in models.items():
    if name in ["K-Nearest Neighbors","Ridge Regression","Linear Regression"]:
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    results.append({"Model": name, "MAE": round(mae,3), "RMSE": round(rmse,3), "R2": round(r2,4)})
    preds_dict[name] = pred

results_df = pd.DataFrame(results).sort_values("RMSE")
results_df.to_csv("models/model_comparison.csv", index=False)

best_model_name = results_df.iloc[0]["Model"]

# comparison chart
fig, ax = plt.subplots(figsize=(8,4))
ax.bar(results_df["Model"], results_df["RMSE"], color="#D8823B")
ax.set_ylabel("RMSE (lower is better)")
ax.set_title("Model Comparison - RMSE")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("visuals/model_comparison_rmse.png", dpi=110)
plt.close()

# actual vs predicted for best model
best_pred = preds_dict[best_model_name]
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(y_test.values[:200], label="Actual", linewidth=1)
ax.plot(best_pred[:200], label=f"Predicted ({best_model_name})", linewidth=1, alpha=0.8)
ax.set_title("Actual vs Predicted PM2.5 (first 200 test hours)")
ax.legend()
plt.tight_layout()
plt.savefig("visuals/actual_vs_predicted.png", dpi=110)
plt.close()

# feature importance (if RF is available)
if "Random Forest (small)" in models:
    rf = models["Random Forest (small)"]
    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7,4))
    importances.plot(kind="barh", ax=ax, color="#3B9C8C")
    ax.invert_yaxis()
    ax.set_title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig("visuals/feature_importance.png", dpi=110)
    plt.close()

summary = {
    "cleaning_log": cleaning_log,
    "best_model": best_model_name,
    "results": results
}
with open("models/summary.json","w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
