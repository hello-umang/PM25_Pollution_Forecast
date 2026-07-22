import pandas as pd
import numpy as np

np.random.seed(42)

# Simulate 1 year of hourly data for 2 stations in California (EPA AQS style)
stations = ["06-037-1103", "06-037-4004"]  # Sample CA county-site codes
date_range = pd.date_range("2023-01-01", "2023-12-31 23:00", freq="h")

rows = []
for station in stations:
    base = 12 if station == "06-037-1103" else 9
    n = len(date_range)
    hour = date_range.hour
    month = date_range.month
    # seasonal pattern: higher in winter, lower in summer
    seasonal = 6 * np.cos((month - 1) / 12 * 2 * np.pi)
    # diurnal pattern: higher in morning/evening rush hours
    diurnal = 3 * np.sin((hour - 6) / 24 * 2 * np.pi) + 2 * (np.isin(hour, [7,8,18,19])).astype(float)
    noise = np.random.normal(0, 2.5, n)
    pm25 = base + seasonal + diurnal + noise
    pm25 = np.clip(pm25, 0, None)

    temp = 15 + 10*np.cos((month-7)/12*2*np.pi) + np.random.normal(0,2,n)
    humidity = 55 + 15*np.sin((month-3)/12*2*np.pi) + np.random.normal(0,5,n)
    wind_speed = np.abs(np.random.normal(3.5, 1.5, n))

    df = pd.DataFrame({
        "station_id": station,
        "datetime": date_range,
        "pm25": pm25.round(2),
        "temperature_c": temp.round(1),
        "humidity_pct": humidity.round(1),
        "wind_speed_mps": wind_speed.round(2),
    })
    rows.append(df)

data = pd.concat(rows, ignore_index=True)

# introduce some missing values & duplicates to make cleaning step meaningful
missing_idx = data.sample(frac=0.01, random_state=1).index
data.loc[missing_idx, "pm25"] = np.nan
dup_rows = data.sample(frac=0.002, random_state=2)
data = pd.concat([data, dup_rows], ignore_index=True)

data = data.sort_values(["station_id","datetime"]).reset_index(drop=True)
data.to_csv("data/pm25_raw.csv", index=False)
print("Rows generated:", len(data))
print(data.head())
