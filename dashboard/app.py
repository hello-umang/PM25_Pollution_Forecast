"""
US PM2.5 Pollution Forecast Dashboard
Run with: streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PM2.5 Pollution Forecast Dashboard", layout="wide")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.title("US PM2.5 Pollution Forecast Dashboard")
st.caption(
    "Time-series regression forecast of next-hour PM2.5 concentration | EPA AQS-style educational data"
)


@st.cache_data
def load_data():
    return pd.read_csv(
        os.path.join(BASE, "data", "pm25_cleaned_features.csv"), parse_dates=["datetime"]
    )


@st.cache_data
def load_model_results():
    return pd.read_csv(os.path.join(BASE, "models", "model_comparison.csv"))


@st.cache_data
def load_summary():
    with open(os.path.join(BASE, "models", "summary.json"), encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_predictions():
    path = os.path.join(BASE, "models", "predictions.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["datetime"])
    return None


try:
    df = load_data()
    results_df = load_model_results()
    summary = load_summary()
    preds = load_predictions()
except FileNotFoundError:
    st.error("Data/model files not found. Run generate_data.py and eda_and_model.py first.")
    st.stop()

# ---- Sidebar ----
st.sidebar.header("Filters")
stations = sorted(df["station_id"].astype(str).unique().tolist())
selected_station = st.sidebar.selectbox("Select Station", stations)
station_df = df[df["station_id"].astype(str) == selected_station].sort_values("datetime")

# ---- KPI Row ----
best_name = summary.get("best_model", results_df.iloc[0]["Model"])
best_row = results_df.iloc[0]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Best Model", str(best_name)[:28])
col2.metric("Test RMSE", f'{best_row["RMSE"]:.3f}')
col3.metric("Test MAE", f'{best_row["MAE"]:.3f}')
col4.metric("Test R²", f'{best_row["R2"]:.3f}')

st.divider()

latest = station_df.iloc[-1]
risk = latest.get("pollution_risk", "N/A")
risk_colors = {
    "Good": "🟢",
    "Moderate": "🟡",
    "Unhealthy(SG)": "🟠",
    "Unhealthy": "🔴",
}
st.subheader(f"Current Status — Station {selected_station}")
st.markdown(
    f"### {risk_colors.get(risk, '')} Pollution Risk: **{risk}**  |  "
    f"Latest PM2.5: **{latest['pm25']:.1f} µg/m³**"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Trend",
        "📊 Model Comparison",
        "🧠 Feature Importance",
        "🔬 Error Analysis",
        "🏙️ Station Comparison",
    ]
)

with tab1:
    st.subheader("PM2.5 Trend (last 500 hours)")
    plot_df = station_df.tail(500)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df["datetime"], plot_df["pm25"], color="#3B7DD8")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title(f"Recent PM2.5 readings — {selected_station}")
    st.pyplot(fig)
    plt.close(fig)

    img_path = os.path.join(BASE, "visuals", "actual_vs_predicted.png")
    if os.path.exists(img_path):
        st.image(img_path, caption="Actual vs Predicted next-hour PM2.5 (test set, best model)")

    if preds is not None:
        st.subheader("Latest test-set predictions for selected station")
        psub = preds[preds["station_id"].astype(str) == selected_station].tail(20)
        st.dataframe(psub, use_container_width=True)

with tab2:
    st.subheader("Model Comparison")
    st.dataframe(results_df, use_container_width=True)
    img_path = os.path.join(BASE, "visuals", "model_comparison_rmse.png")
    if os.path.exists(img_path):
        st.image(img_path)
    st.markdown("**Business interpretation**")
    st.write(summary.get("business_interpretation", ""))

with tab3:
    st.subheader("Feature Importance")
    img_path = os.path.join(BASE, "visuals", "feature_importance.png")
    fi_csv = os.path.join(BASE, "models", "feature_importance.csv")
    if os.path.exists(img_path):
        st.image(img_path)
    if os.path.exists(fi_csv):
        st.dataframe(pd.read_csv(fi_csv), use_container_width=True)
    else:
        st.info("Feature importance chart not found for the selected best model type.")

with tab4:
    st.subheader("Error Analysis & Limitations")
    img_path = os.path.join(BASE, "visuals", "error_analysis.png")
    if os.path.exists(img_path):
        st.image(img_path)
    if preds is not None:
        p = preds.copy()
        st.metric("Mean Absolute Error (test predictions file)", f"{p['abs_error'].mean():.3f}")
        st.write("Worst absolute errors (top 10)")
        st.dataframe(p.nlargest(10, "abs_error"), use_container_width=True)
        # Risk classification agreement
        agree = (p["predicted_risk"] == p["actual_risk"]).mean()
        st.metric("Risk-category agreement", f"{agree:.1%}")
    st.markdown("**Limitations**")
    for item in summary.get("limitations", []):
        st.write(f"- {item}")
    st.markdown("**Recommendations**")
    for item in summary.get("recommendations", []):
        st.write(f"- {item}")

with tab5:
    st.subheader("Station Comparison")
    fig, ax = plt.subplots(figsize=(10, 4))
    for st_id in stations:
        sub = (
            df[df["station_id"].astype(str) == st_id]
            .set_index("datetime")["pm25"]
            .resample("D")
            .mean()
        )
        ax.plot(sub.index, sub.values, label=st_id, linewidth=0.9)
    ax.legend()
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("Daily Average PM2.5 by Station")
    st.pyplot(fig)
    plt.close(fig)

    img_path = os.path.join(BASE, "visuals", "daily_pm25_trend.png")
    if os.path.exists(img_path):
        st.image(img_path)

st.divider()
st.caption(
    "Data source pattern: EPA Air Quality System (AQS) — "
    "https://www.epa.gov/aqs/obtaining-aqs-data | Sample educational dataset used for demo."
)
