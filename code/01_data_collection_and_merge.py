# %% [markdown]
# # 01 - Data Collection & Merge
# BreatheSafe - Smog & Asthma Risk Predictor (NUML, ML Lab, Fall 2026)
#
# This script collects four datasets and merges them into one modelling table:
# 1. Global Air Pollution Dataset (Kaggle)
# 2. Air Quality Data in India (Kaggle)
# 3. Asthma Disease Dataset (Kaggle)
# 4. Open-Meteo Air Quality + Weather API (live, no key needed)
#
# If a Kaggle CSV is not in `data/raw/`, or the API cannot be reached, a realistic
# fallback generator is used so the pipeline always runs. Every fallback is logged
# and recorded in `outputs/metrics/dataset_summary.json`.

# %%
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

try:
    HERE = Path(__file__).resolve().parent
except NameError:  # running inside Jupyter
    HERE = Path.cwd()
ROOT = HERE.parent if HERE.name == "code" else HERE
RAW, PROC, METRICS = ROOT / "data/raw", ROOT / "data/processed", ROOT / "outputs/metrics"
for p in (RAW, PROC, METRICS, ROOT / "outputs/plots", ROOT / "models"):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)
summary = {}  # filled as we go, saved at the end

# city: (lat, lon, base PM2.5 for fallback, base temperature for fallback, population weight)
CITIES = {
    "Lahore": (31.5497, 74.3436, 62, 31, 0.30),
    "Islamabad": (33.6844, 73.0479, 38, 28, 0.15),
    "Rawalpindi": (33.5651, 73.0169, 44, 29, 0.15),
    "Karachi": (24.8607, 67.0011, 30, 31, 0.28),
    "Peshawar": (34.0151, 71.5249, 55, 30, 0.12),
}
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_URL = "https://api.open-meteo.com/v1/forecast"

# %% [markdown]
# ## Dataset 4 - Open-Meteo live air quality + weather (5 Pakistani cities)
# Up to 92 past days of hourly data per city, so five cities give roughly 11,000 rows.

# %%
def fetch_open_meteo(city, lat, lon, past_days=92):
    common = dict(latitude=lat, longitude=lon, past_days=past_days, forecast_days=1, timezone="Asia/Karachi")
    aq = requests.get(AQ_URL, params={**common, "hourly": "pm10,pm2_5,nitrogen_dioxide,sulphur_dioxide,ozone,carbon_monoxide"}, timeout=30)
    aq.raise_for_status()
    wx = requests.get(WX_URL, params={**common, "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m"}, timeout=30)
    wx.raise_for_status()
    df = pd.DataFrame(aq.json()["hourly"]).merge(pd.DataFrame(wx.json()["hourly"]), on="time")
    df = df.rename(columns={"time": "timestamp", "nitrogen_dioxide": "no2", "sulphur_dioxide": "so2", "ozone": "o3",
                            "carbon_monoxide": "co", "temperature_2m": "temp", "relative_humidity_2m": "humidity",
                            "wind_speed_10m": "wind"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["city"] = city
    return df[df["timestamp"] <= pd.Timestamp.now()]  # drop forecast hours


def synthetic_timeline():
    """Fallback: hourly smog + weather for 5 cities, Sep-Nov 2025, smog building toward November."""
    idx = pd.date_range("2025-09-01", periods=2208, freq="h")
    n = len(idx)
    hod, prog = idx.hour.values, np.arange(n) / n
    frames = []
    for city, (lat, lon, base, tbase, _) in CITIES.items():
        season = 0.55 + 1.9 * prog ** 1.6
        diurnal = 1 + 0.28 * np.cos((hod - 5) / 24 * 2 * np.pi)
        ar = np.zeros(n)
        for i in range(1, n):
            ar[i] = 0.96 * ar[i - 1] + rng.normal(0, 0.09)
        episode = np.zeros(n)
        for s in rng.choice(n - 100, 8, replace=False):
            episode[s:s + rng.integers(36, 96)] += rng.uniform(0.3, 0.9)
        pm25 = base * season * diurnal * np.exp(ar + 0.6 * episode)
        cooling = 5 if city == "Karachi" else 12
        temp = tbase - cooling * prog + 7 * np.sin((hod - 9) / 24 * 2 * np.pi) + rng.normal(0, 1.2, n)
        hum = np.clip(55 + 20 * prog - 1.2 * (temp - 22) + rng.normal(0, 6, n), 10, 100)
        wind = rng.gamma(2.0, 4.0, n) * np.where(episode > 0.3, 0.5, 1.0)
        df = pd.DataFrame({
            "timestamp": idx, "city": city, "pm2_5": pm25,
            "pm10": pm25 * rng.uniform(1.5, 2.3, n) + rng.normal(0, 6, n),
            "no2": 18 + 0.12 * pm25 + rng.normal(0, 5, n),
            "so2": 6 + 0.05 * pm25 + rng.normal(0, 2, n),
            "o3": np.clip(40 + 35 * np.sin((hod - 9) / 24 * 2 * np.pi) - 0.05 * pm25 + rng.normal(0, 8, n), 5, None),
            "co": 300 + 3 * pm25 + rng.normal(0, 40, n),
            "temp": temp, "humidity": hum, "wind": wind,
        })
        # typical field faults: negative readings, spikes, and gaps
        cols = ["pm2_5", "pm10", "no2", "so2", "o3", "co", "temp", "humidity", "wind"]
        neg = rng.choice(n, int(0.003 * n), replace=False)
        df.loc[neg, "pm2_5"] *= -1
        spk = rng.choice(n, int(0.005 * n), replace=False)
        df.loc[spk, "pm2_5"] *= 6
        df.loc[spk, "pm10"] *= 3
        for _ in range(4):
            s = rng.integers(30, n - 30)
            df.loc[s:s + rng.integers(3, 10), cols] = np.nan
        for c in cols:
            df.loc[rng.random(n) < 0.005, c] = np.nan
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


try:
    timeline = pd.concat([fetch_open_meteo(c, v[0], v[1]) for c, v in CITIES.items()], ignore_index=True)
    if len(timeline) < 10000:
        raise ValueError(f"only {len(timeline)} live rows")
    ds4_mode = "live Open-Meteo API"
except Exception as exc:  # no internet, API down, firewall...
    print(f"[WARN] Open-Meteo unavailable ({type(exc).__name__}: {exc}). Using synthetic Open-Meteo-style timeline.")
    timeline = synthetic_timeline()
    ds4_mode = "FALLBACK synthetic (Open-Meteo format)"

timeline = timeline.sort_values(["city", "timestamp"]).reset_index(drop=True)
timeline.to_csv(PROC / "environment_timeline.csv", index=False)
print(f"Dataset 4 [{ds4_mode}]: {len(timeline):,} hourly records, {timeline.city.nunique()} cities")
summary["dataset_4_open_meteo"] = {"records": int(len(timeline)), "mode": ds4_mode}

# %% [markdown]
# ## Dataset 2 - Air Quality Data in India (Kaggle `city_day.csv`)

# %%
def synthetic_india_city_day():
    cities = ["Delhi", "Lucknow", "Kolkata", "Patna", "Ahmedabad", "Mumbai", "Chennai", "Bengaluru",
              "Hyderabad", "Jaipur", "Amritsar", "Chandigarh", "Gurugram", "Visakhapatnam", "Bhopal"]
    base = dict(zip(cities, [115, 105, 80, 110, 70, 50, 40, 40, 50, 75, 80, 70, 100, 55, 65]))
    dates = pd.date_range("2015-01-01", "2020-07-01", freq="D")
    month_f = {1: 1.6, 2: 1.2, 3: 0.9, 4: 0.85, 5: 0.9, 6: 0.7, 7: 0.45, 8: 0.4, 9: 0.55, 10: 1.0, 11: 1.7, 12: 1.9}
    rows = []
    for c in cities:
        f = np.array([month_f[m] for m in dates.month])
        pm25 = base[c] * f * np.exp(rng.normal(0, 0.35, len(dates)))
        rows.append(pd.DataFrame({
            "City": c, "Date": dates, "PM2.5": pm25, "PM10": pm25 * rng.uniform(1.5, 2.2, len(dates)),
            "NO": rng.gamma(2, 6, len(dates)), "NO2": 20 + 0.15 * pm25 + rng.normal(0, 8, len(dates)),
            "NOx": 30 + 0.2 * pm25 + rng.normal(0, 10, len(dates)), "NH3": rng.gamma(3, 6, len(dates)),
            "CO": np.abs(0.6 + 0.008 * pm25 + rng.normal(0, 0.3, len(dates))),
            "SO2": np.abs(8 + 0.03 * pm25 + rng.normal(0, 3, len(dates))),
            "O3": np.abs(35 + rng.normal(0, 15, len(dates))),
            "AQI": np.clip(pm25 * 1.6 + rng.normal(0, 15, len(dates)), 10, 500)}))
    df = pd.concat(rows, ignore_index=True)
    for c in ["PM2.5", "PM10", "NO2", "SO2", "O3", "CO"]:
        df.loc[rng.random(len(df)) < 0.12, c] = np.nan
    return df.sample(n=29531, random_state=SEED).sort_values(["City", "Date"]).reset_index(drop=True)


f2 = next(iter(sorted(RAW.glob("city_day*.csv"))), None)
if f2 is not None:
    india = pd.read_csv(f2, parse_dates=["Date"])
    ds2_mode = f"REAL Kaggle file ({f2.name})"
else:
    india = synthetic_india_city_day()
    ds2_mode = "FALLBACK synthetic (city_day format)"
keep = [c for c in ["City", "Date", "PM2.5", "PM10", "NO2", "SO2", "O3", "CO", "AQI"] if c in india.columns]
india = india[keep]
india.to_csv(PROC / "india_city_day_standardized.csv", index=False)
print(f"Dataset 2 [{ds2_mode}]: {len(india):,} city-day records, {india.City.nunique()} cities")
summary["dataset_2_india_air_quality"] = {"records": int(len(india)), "mode": ds2_mode}

# %% [markdown]
# ## Dataset 1 - Global Air Pollution Dataset (Kaggle)

# %%
def synthetic_global_air():
    n = 23500
    pm = np.clip(rng.lognormal(3.6, 0.9, n), 1, 500)
    df = pd.DataFrame({
        "Country": rng.choice(["Pakistan", "India", "China", "USA", "Brazil", "Germany", "Egypt", "Nigeria", "Japan", "Iran"], n),
        "City": [f"City_{i}" for i in rng.integers(0, 4000, n)],
        "PM2.5 AQI Value": pm.round(),
        "CO AQI Value": np.clip(rng.gamma(1.5, 1.2, n), 0, 100).round(),
        "Ozone AQI Value": np.clip(rng.normal(38, 22, n), 0, 250).round(),
        "NO2 AQI Value": np.clip(rng.gamma(1.4, 2.0, n), 0, 100).round()})
    df["AQI Value"] = df[["PM2.5 AQI Value", "CO AQI Value", "Ozone AQI Value", "NO2 AQI Value"]].max(axis=1)
    df["AQI Category"] = pd.cut(df["AQI Value"], [-1, 50, 100, 150, 200, 300, 1000],
                                labels=["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"])
    return df


f1 = next(iter(sorted(RAW.glob("*global*pollution*.csv"))), None)
if f1 is not None:
    glob_air = pd.read_csv(f1)
    ds1_mode = f"REAL Kaggle file ({f1.name})"
else:
    glob_air = synthetic_global_air()
    ds1_mode = "FALLBACK synthetic (global pollution format)"
glob_air.to_csv(PROC / "global_air_pollution_standardized.csv", index=False)
print(f"Dataset 1 [{ds1_mode}]: {len(glob_air):,} records")
summary["dataset_1_global_air_pollution"] = {"records": int(len(glob_air)), "mode": ds1_mode}

# %% [markdown]
# ## Dataset 3 - Asthma Disease Dataset (Kaggle)
# The real Kaggle file is small (a few thousand patients). If it has fewer than 10,000 rows, the pool is
# topped up by bootstrap-with-jitter and the extra rows are flagged `is_augmented = 1`.

# %%
CLIN_COLS = ["Age", "BMI", "Smoking", "PhysicalActivity", "PollutionExposure", "DustExposure", "PetAllergy",
             "FamilyHistoryAsthma", "HistoryOfAllergies", "LungFunctionFEV1", "Wheezing", "ShortnessOfBreath",
             "ChestTightness", "Coughing", "NighttimeSymptoms", "ExerciseInduced"]


def synthetic_patients(n=10200):
    age = rng.integers(5, 82, n)
    lat = rng.normal(0, 1, n)
    df = pd.DataFrame({
        "Age": age, "Gender": rng.integers(0, 2, n),
        "BMI": np.clip(rng.normal(26.5, 6, n), 14, 45),
        "Smoking": ((rng.random(n) < 0.16) & (age > 17)).astype(int),
        "PhysicalActivity": rng.uniform(0, 10, n), "PollutionExposure": rng.uniform(0, 10, n),
        "DustExposure": rng.uniform(0, 10, n), "PetAllergy": (rng.random(n) < 0.2).astype(int),
        "FamilyHistoryAsthma": (rng.random(n) < 0.25).astype(int), "HistoryOfAllergies": (rng.random(n) < 0.3).astype(int),
        "LungFunctionFEV1": np.clip(3.3 - 0.018 * np.abs(age - 30) + rng.normal(0, 0.55, n) - 0.3 * lat * 0, 1.0, 4.2)})
    for c, w in [("Wheezing", .6), ("ShortnessOfBreath", .6), ("ChestTightness", .5), ("Coughing", .5),
                 ("NighttimeSymptoms", .4), ("ExerciseInduced", .4)]:
        p = 1 / (1 + np.exp(-(-0.9 + w * lat + 0.4 * df["FamilyHistoryAsthma"] + 0.3 * df["Smoking"])))
        df[c] = (rng.random(n) < p).astype(int)
    df["Diagnosis"] = (df[["Wheezing", "ShortnessOfBreath", "ChestTightness", "Coughing"]].sum(axis=1) + rng.normal(0, .8, n) > 2).astype(int)
    for c in ["BMI", "LungFunctionFEV1", "PollutionExposure", "DustExposure", "PhysicalActivity"]:
        df.loc[rng.random(n) < 0.03, c] = np.nan  # missing clinical entries
    return df


f3 = next(iter(sorted(RAW.glob("asthma*.csv"))), None)
patients_real_rows = 0
if f3 is not None:
    pat = pd.read_csv(f3)
    patients_real_rows = len(pat)
    for c in CLIN_COLS:
        if c not in pat.columns:
            pat[c] = np.nan
    if "Gender" not in pat.columns:
        pat["Gender"] = np.nan
    if "Diagnosis" not in pat.columns:
        pat["Diagnosis"] = np.nan
    pat = pat[["Age", "Gender"] + [c for c in CLIN_COLS if c != "Age"] + ["Diagnosis"]].copy()
    pat["is_augmented"] = 0
    empty_cols = [c for c in CLIN_COLS if pat[c].isna().all()]
    if empty_cols:  # e.g. a different asthma CSV that lacks some clinical fields
        print(f"[WARN] Columns missing in the real asthma file, filled from the fallback generator: {empty_cols}")
        filler = synthetic_patients(len(pat))
        for c in empty_cols:
            pat[c] = filler[c].values
    ds3_mode = f"REAL Kaggle file ({f3.name}), {patients_real_rows:,} rows"
    if len(pat) < 10000:
        extra = pat.sample(n=10000 - len(pat), replace=True, random_state=SEED).copy()
        extra["Age"] = np.clip(extra["Age"] + rng.integers(-2, 3, len(extra)), 1, 90)
        extra["BMI"] = np.clip(extra["BMI"] + rng.normal(0, 0.8, len(extra)), 12, 50)
        extra["LungFunctionFEV1"] = np.clip(extra["LungFunctionFEV1"] + rng.normal(0, 0.08, len(extra)), 0.8, 4.5)
        extra["is_augmented"] = 1
        pat = pd.concat([pat, extra], ignore_index=True)
        print(f"[WARN] Real asthma file has {patients_real_rows:,} rows (<10,000). Added {len(extra):,} bootstrap+jitter rows (is_augmented=1).")
else:
    pat = synthetic_patients()
    pat["is_augmented"] = 0
    ds3_mode = "FALLBACK synthetic (asthma dataset format)"
pat.insert(0, "patient_id", np.arange(len(pat)))
pat.to_csv(PROC / "patients_pool.csv", index=False)
print(f"Dataset 3 [{ds3_mode}]: {len(pat):,} patient records")
summary["dataset_3_asthma"] = {"records": int(len(pat)), "mode": ds3_mode, "real_rows": int(patients_real_rows)}

# %% [markdown]
# ## Join strategy (spatiotemporal + clinical risk tier)
# 1. **Spatial key.** Each patient gets a home city from the five Open-Meteo cities (population-weighted draw, stored once per patient).
# 2. **Temporal key.** A patient-day record is created by pairing a patient with a random hour from *their* city's Open-Meteo timeline (rounded to the hour).
# 3. **Environmental context.** The exact pollutant + weather values at that city-hour are attached. Datasets 1 and 2 supply the AQI reference table and the multi-year seasonal baseline used in the EDA.
# 4. **Clinical risk tier.** Patient clinical profile + ambient exposure produce a four-level label (Low / Moderate / High / Critical).
#
# **Important:** no public dataset links a person's daily pollution exposure to an asthma attack. The label is therefore a documented
# *proxy* built from epidemiological relationships (PM2.5, gases, weather, age, smoking, FEV1, symptoms, allergy) plus noise.

# %%
N_RECORDS, N_DUP = 15000, 75
city_names = list(CITIES)
city_w = np.array([CITIES[c][4] for c in city_names])
pat["city"] = rng.choice(city_names, size=len(pat), p=city_w / city_w.sum())

pick = rng.integers(0, len(pat), N_RECORDS)
recs = pat.iloc[pick].reset_index(drop=True).copy()
stamps = []
for city in city_names:
    tl = timeline[timeline.city == city].sort_values("timestamp").reset_index(drop=True)
    tl = tl.iloc[24:]  # keep rows that have >= 24h history for lag/rolling features
    m = (recs["city"] == city).values
    stamps.append(pd.Series(tl["timestamp"].values[rng.integers(0, len(tl), m.sum())], index=recs.index[m]))
recs["timestamp"] = pd.concat(stamps).sort_index()
env_cols = ["pm2_5", "pm10", "no2", "so2", "o3", "co", "temp", "humidity", "wind"]
merged = recs.merge(timeline[["city", "timestamp"] + env_cols], on=["city", "timestamp"], how="left")

# %% [markdown]
# ## Label creation (proxy asthma-risk tier)

# %%
def make_risk_label(df):
    d = df.copy()
    for c in env_cols + CLIN_COLS:
        d[c] = d[c].fillna(d[c].median())  # only for scoring; the real table keeps its gaps
    pm = d.pm2_5.clip(0, 400) / 60
    gas = 0.5 * d.no2.clip(0, 300) / 80 + 0.5 * d.so2.clip(0, 200) / 40
    stagn = (d.humidity / 100) / (d.wind + 1)
    age_risk = ((d.Age <= 12) | (d.Age >= 60)).astype(float)
    clinical = (0.9 * d.FamilyHistoryAsthma + 0.8 * d.Smoking + 0.6 * d.HistoryOfAllergies + 0.35 * d.PetAllergy
                + 0.12 * d.DustExposure + 0.12 * d.PollutionExposure + 0.9 * age_risk
                + 0.8 * (4 - d.LungFunctionFEV1).clip(0, 3) / 3 + 0.35 * d.Wheezing + 0.35 * d.ShortnessOfBreath
                + 0.3 * d.ChestTightness + 0.2 * d.Coughing + 0.35 * d.NighttimeSymptoms + 0.4 * d.ExerciseInduced)
    score = (1.6 * pm + 0.5 * gas + 0.25 * d.o3 / 100 + 0.35 * stagn * 10 + clinical
             + 0.8 * pm * (age_risk + d.FamilyHistoryAsthma) + 0.04 * d.PhysicalActivity * pm
             + rng.normal(0, 0.6, len(d)))
    q = np.quantile(score, [0.35, 0.68, 0.90])
    return np.digitize(score, q)


merged["Asthma_Risk_Level"] = make_risk_label(merged)
merged["Asthma_Risk_Label"] = merged["Asthma_Risk_Level"].map({0: "Low", 1: "Moderate", 2: "High", 3: "Critical"})
merged.insert(0, "record_id", np.arange(len(merged)))

# repeated submissions / feed retries: a few exact duplicates so the cleaning step has real work to do
dups = merged.sample(N_DUP, random_state=SEED)
merged = pd.concat([merged, dups], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
merged = merged.drop(columns=["Diagnosis"], errors="ignore")
merged.to_csv(PROC / "merged_breathesafe_dataset.csv", index=False)
print(f"Merged dataset: {merged.shape[0]:,} rows x {merged.shape[1]} cols -> data/processed/merged_breathesafe_dataset.csv")
print(merged["Asthma_Risk_Label"].value_counts(normalize=True).round(3).to_string())

# %% [markdown]
# ## Documentation: sources, bias, join strategy

# %%
summary["merged_rows"] = int(len(merged))
summary["target_distribution"] = merged["Asthma_Risk_Label"].value_counts().to_dict()
with open(METRICS / "dataset_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)

doc = f"""# Data sources, bias and join strategy

## Sources
| # | Dataset | URL | Records used | Mode this run |
|---|---------|-----|-------------:|---------------|
| 1 | Global Air Pollution Dataset | https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset | {summary['dataset_1_global_air_pollution']['records']:,} | {ds1_mode} |
| 2 | Air Quality Data in India | https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india | {summary['dataset_2_india_air_quality']['records']:,} | {ds2_mode} |
| 3 | Asthma Disease Dataset | https://www.kaggle.com/datasets/rabieelkharoua/asthma-disease-dataset | {summary['dataset_3_asthma']['records']:,} | {ds3_mode} |
| 4 | Open-Meteo Air Quality + Weather API | https://air-quality-api.open-meteo.com/v1/air-quality , https://api.open-meteo.com/v1/forecast | {summary['dataset_4_open_meteo']['records']:,} | {ds4_mode} |

## Known biases
- **Urban sensor skew:** monitoring stations sit mostly in large cities; rural exposure is under-represented.
- **Winter smog seasonality:** PM2.5 peaks October-January in the Indo-Gangetic plain; a three-month window over-weights one season.
- **Self-reported clinical bias:** symptoms, exposure and allergy fields are self-reported and can be under- or over-stated.
- **Cross-country transfer:** the Indian and global datasets are used for seasonal context, not as a direct measurement of Pakistani exposure.
- **Proxy label:** the risk tier is rule-based plus noise, not a recorded clinical outcome.

## Join strategy
Spatial key (city) + temporal key (hour) link a patient-day to the Open-Meteo timeline; Datasets 1-2 provide AQI reference and seasonal baselines;
the clinical risk tier combines patient profile and ambient exposure. See the notebook markdown above for the full description.
"""
(METRICS / "data_sources_bias_join.md").write_text(doc, encoding="utf-8")
print("Saved outputs/metrics/dataset_summary.json and data_sources_bias_join.md")
