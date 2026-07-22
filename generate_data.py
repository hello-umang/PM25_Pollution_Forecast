"""
Generate an EPA AQS-style hourly PM2.5 sample dataset for offline execution.

Official source pattern:
  United States EPA Air Quality System (AQS)
  https://www.epa.gov/aqs/obtaining-aqs-data
  Parameter code 88101 (PM2.5 Local Conditions)

This script creates a synthetic educational dataset that mirrors the hourly
schema used in the rest of the project so the pipeline runs without API keys.
"""

from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Two California-style monitoring station IDs (county-site style codes)
STATIONS = ["60371103", "60650005"]
DATE_RANGE = pd.date_range("2026-01-01", "2026-12-31 23:00", freq="h")

rows = []
for station in STATIONS:
    base = 12.0 if station == "60371103" else 9.0
    n = len(DATE_RANGE)
    hour = DATE_RANGE.hour.to_numpy()
    month = DATE_RANGE.month.to_numpy()
    dayofweek = DATE_RANGE.dayofweek.to_numpy()

    # Seasonal: higher in winter, lower in summer
    seasonal = 6.0 * np.cos((month - 1) / 12.0 * 2.0 * np.pi)
    # Diurnal: elevated near morning/evening traffic peaks
    diurnal = (
        3.0 * np.sin((hour - 6) / 24.0 * 2.0 * np.pi)
        + 2.0 * np.isin(hour, [7, 8, 18, 19]).astype(float)
    )
    # Mild weekday effect
    weekday_effect = np.where(dayofweek < 5, 0.8, -0.6)
    noise = np.random.normal(0, 2.5, n)
    pm25 = np.clip(base + seasonal + diurnal + weekday_effect + noise, 0, None)

    temperature_c = (
        15.0
        + 10.0 * np.cos((month - 7) / 12.0 * 2.0 * np.pi)
        + np.random.normal(0, 2.0, n)
    )
    humidity_pct = (
        55.0
        + 15.0 * np.sin((month - 3) / 12.0 * 2.0 * np.pi)
        + np.random.normal(0, 5.0, n)
    )
    wind_speed_mps = np.abs(np.random.normal(3.5, 1.5, n))

    # Weak physical relationships: cooler / calmer / more humid -> slightly higher PM
    pm25 = pm25 + 0.08 * (20 - temperature_c) + 0.05 * (humidity_pct - 50) - 0.4 * wind_speed_mps
    pm25 = np.clip(pm25, 0, None)

    rows.append(
        pd.DataFrame(
            {
                "station_id": station,
                "datetime": DATE_RANGE,
                "pm25": np.round(pm25, 2),
                "temperature_c": np.round(temperature_c, 1),
                "humidity_pct": np.round(humidity_pct, 1),
                "wind_speed_mps": np.round(wind_speed_mps, 2),
            }
        )
    )

data = pd.concat(rows, ignore_index=True)

# Introduce missing values and duplicates so cleaning steps are meaningful
missing_idx = data.sample(frac=0.01, random_state=1).index
data.loc[missing_idx, "pm25"] = np.nan
dup_rows = data.sample(frac=0.002, random_state=2)
data = pd.concat([data, dup_rows], ignore_index=True)

data = data.sort_values(["station_id", "datetime"]).reset_index(drop=True)
out_path = OUT_DIR / "pm25_raw.csv"
data.to_csv(out_path, index=False)

print(f"Rows generated: {len(data)}")
print(f"Stations: {data['station_id'].nunique()}")
print(f"Date range: {data['datetime'].min()} -> {data['datetime'].max()}")
print(f"Missing pm25: {data['pm25'].isna().sum()}")
print(f"Saved: {out_path}")
print(data.head())
