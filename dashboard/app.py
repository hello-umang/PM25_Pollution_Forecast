"""
US PM2.5 Pollution Forecast Dashboard
Run with: streamlit run dashboard/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

st.set_page_config(page_title="PM2.5 Pollution Forecast Dashboard", layout="wide")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.title("🌫️ US PM2.5 Pollution Forecast Dashboard")
st.caption("Time-series regression forecast of next-hour PM2.5 concentration | EPA AQS-style data")

# ---- Load data ----
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE, "data", "pm25_cleaned_features.csv"), parse_dates=["datetime"])
    return df

@st.cache_data
def load_model_results():
    return pd.read_csv(os.path.join(BASE, "models", "model_comparison.csv"))

@st.cache_data
def load_summary():
    with open(os.path.join(BASE, "models", "summary.json")) as f:
        return json.load(f)

try:
    df = load_data()
    results_df = load_model_results()
    summary = load_summary()
except FileNotFoundError:
    st.error("Data/model files not found. Run generate_data.py and eda_and_model.py first.")
    st.stop()

# ---- Sidebar ----
st.sidebar.header("Filters")
stations = df["station_id"].unique().tolist()
selected_station = st.sidebar.selectbox("Select Station", stations)

station_df = df[df["station_id"] == selected_station].sort_values("datetime")

# ---- KPI Row ----
col1, col2, col3, col4 = st.columns(4)
col1.metric("Best Model", summary["best_model"])
best_row = results_df.iloc[0]
col2.metric("Best Model RMSE", f'{best_row["RMSE"]:.2f}')
col3.metric("Best Model MAE", f'{best_row["MAE"]:.2f}')
col4.metric("Best Model R²", f'{best_row["R2"]:.3f}')

st.divider()

# ---- Pollution Risk Category ----
latest = station_df.iloc[-1]
risk = latest["pollution_risk"]
risk_colors = {"Good": "🟢", "Moderate": "🟡", "Unhealthy(SG)": "🟠", "Unhealthy": "🔴"}
st.subheader(f"Current Status — Station {selected_station}")
st.markdown(f"### {risk_colors.get(risk,'')} Pollution Risk: **{risk}**  |  Latest PM2.5: **{latest['pm25']:.1f} µg/m³**")

# ---- Actual vs Predicted-style trend (using stored image) ----
tab1, tab2, tab3, tab4 = st.tabs(["📈 Trend", "📊 Model Comparison", "🧮 Feature Importance", "🗂️ Station Comparison"])

with tab1:
    st.subheader("PM2.5 Trend (last 500 hours)")
    plot_df = station_df.tail(500)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df["datetime"], plot_df["pm25"], color="#3B7DD8")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title(f"Recent PM2.5 readings — {selected_station}")
    st.pyplot(fig)

    img_path = os.path.join(BASE, "visuals", "actual_vs_predicted.png")
    if os.path.exists(img_path):
        st.image(img_path, caption="Actual vs Predicted PM2.5 (test set, best model)")

with tab2:
    st.subheader("Model Comparison")
    st.dataframe(results_df, use_container_width=True)
    img_path = os.path.join(BASE, "visuals", "model_comparison_rmse.png")
    if os.path.exists(img_path):
        st.image(img_path)

with tab3:
    st.subheader("Feature Importance (Random Forest)")
    img_path = os.path.join(BASE, "visuals", "feature_importance.png")
    if os.path.exists(img_path):
        st.image(img_path)
    else:
        st.info("Feature importance chart not found.")

with tab4:
    st.subheader("Station Comparison")
    fig, ax = plt.subplots(figsize=(10, 4))
    for st_id in stations:
        sub = df[df["station_id"] == st_id].set_index("datetime")["pm25"].resample("D").mean()
        ax.plot(sub.index, sub.values, label=st_id, linewidth=0.8)
    ax.legend()
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("Daily Average PM2.5 by Station")
    st.pyplot(fig)

st.divider()
st.caption("Data source: EPA Air Quality System (AQS) — https://www.epa.gov/aqs/obtaining-aqs-data | Sample data used for demo purposes.")
