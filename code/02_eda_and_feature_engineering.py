# %% [markdown]
# # 02 - Cleaning, Imputation, Outliers, Feature Engineering, Feature Selection, EDA plots
# Input: `data/processed/merged_breathesafe_dataset.csv` + `environment_timeline.csv` (from script 01)
# Output: `data/processed/engineered_features.csv`, `outputs/plots/fig01..fig08`, `outputs/metrics/*`
# (Figures 09-12 need trained models, so script 03 saves them.)

# %%
import json
import sys
import warnings
from functools import partial
from pathlib import Path

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
ROOT = HERE.parent if HERE.name == "code" else HERE
PROC, PLOTS, METRICS = ROOT / "data/processed", ROOT / "outputs/plots", ROOT / "outputs/metrics"
for p in (PROC, PLOTS, METRICS):
    p.mkdir(parents=True, exist_ok=True)
SEED = 42
sns.set_theme(style="whitegrid", context="notebook")
CLASS_NAMES = ["Low", "Moderate", "High", "Critical"]
interp = {}   # figure -> written interpretation


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS / name, dpi=200, bbox_inches="tight")
    if "ipykernel" in sys.modules:
        plt.show()
    plt.close(fig)


# %% [markdown]
# ## 1. Load + data cleaning (duplicates, timestamp alignment, invalid sensor values)

# %%
merged = pd.read_csv(PROC / "merged_breathesafe_dataset.csv", parse_dates=["timestamp"])
timeline = pd.read_csv(PROC / "environment_timeline.csv", parse_dates=["timestamp"])
log = {"rows_raw": int(len(merged))}

merged = merged.drop_duplicates().reset_index(drop=True)
log["duplicates_removed"] = log["rows_raw"] - len(merged)

# timestamp alignment: everything on the hour, one reading per city-hour
merged["timestamp"] = merged["timestamp"].dt.floor("h")
timeline["timestamp"] = timeline["timestamp"].dt.floor("h")
n0 = len(timeline)
timeline = timeline.drop_duplicates(["city", "timestamp"]).sort_values(["city", "timestamp"]).reset_index(drop=True)
log["timeline_duplicate_hours_removed"] = n0 - len(timeline)

POLL = ["pm2_5", "pm10", "no2", "so2", "o3", "co"]
ENV = POLL + ["temp", "humidity", "wind"]
invalid = {}
for c in POLL + ["wind"]:
    bad = timeline[c] < 0
    invalid[c] = int(bad.sum())
    timeline.loc[bad, c] = np.nan
bad = (timeline["humidity"] < 0) | (timeline["humidity"] > 100)
invalid["humidity"] = int(bad.sum())
timeline.loc[bad, "humidity"] = np.nan
log["invalid_sensor_values_set_to_nan"] = invalid

# %% [markdown]
# ## 2. Environmental sensors: time-series interpolation
# Short gaps (<= 6 h) are interpolated in time; longer gaps are filled from the nearest valid reading.

# %%
log["timeline_missing_before"] = {c: int(timeline[c].isna().sum()) for c in ENV}
parts = []
for city, g in timeline.groupby("city"):
    g = g.set_index("timestamp").sort_index()
    g[ENV] = g[ENV].interpolate(method="time", limit=6, limit_direction="both").ffill().bfill()
    parts.append(g.reset_index())
timeline = pd.concat(parts, ignore_index=True)
log["timeline_missing_after"] = {c: int(timeline[c].isna().sum()) for c in ENV}
pm25_before = timeline["pm2_5"].copy()   # saved for the before/after plot

# %% [markdown]
# ## 3. Outlier handling: IQR winsorization
# Pollutants use a wide fence (Q3 + 3 x IQR, floor at 0) so real severe smog episodes survive; only sensor-spike
# territory is capped. Weather variables use the usual 1.5 x IQR fence on both sides.

# %%
def winsorize(s, k, lower=True):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo = q1 - k * iqr if lower else 0
    hi = q3 + k * iqr
    return s.clip(lo, hi), int(((s < lo) | (s > hi)).sum()), float(hi)


capped = {}
for c in POLL:
    timeline[c], capped[c], hi = winsorize(timeline[c], 3.0, lower=False)
    capped[c] = {"n_capped": capped[c], "upper_fence": round(hi, 2)}
for c in ["temp", "humidity", "wind"]:
    timeline[c], n, hi = winsorize(timeline[c], 1.5)
    capped[c] = {"n_capped": n, "upper_fence": round(hi, 2)}
timeline["humidity"] = timeline["humidity"].clip(0, 100)
log["environment_capped"] = capped

# %% [markdown]
# ## 4. Time-based features from the cleaned timeline (rolling 24 h, 1-day lag, daily temperature range)

# %%
g = timeline.groupby("city")
timeline["pm25_roll24"] = g["pm2_5"].transform(lambda s: s.rolling(24, min_periods=12).mean())
timeline["pm25_lag1d"] = g["pm2_5"].shift(24)
timeline["temp_range24"] = g["temp"].transform(lambda s: s.rolling(24, min_periods=12).max() - s.rolling(24, min_periods=12).min())
for c in ["pm25_roll24", "pm25_lag1d", "temp_range24"]:
    timeline[c] = timeline.groupby("city")[c].transform(lambda s: s.bfill())

# weekly city trend for the dashboard / fig03
weekly = (timeline.set_index("timestamp").groupby("city")["pm2_5"].resample("W").mean().reset_index())
weekly.to_csv(METRICS / "city_pm25_weekly.csv", index=False)

# attach cleaned environment to patient-day records (replaces the raw readings)
merged = merged.drop(columns=ENV)
merged = merged.merge(timeline[["city", "timestamp"] + ENV + ["pm25_roll24", "pm25_lag1d", "temp_range24"]],
                      on=["city", "timestamp"], how="left")
log["rows_without_environment_match"] = int(merged["pm2_5"].isna().sum())
merged = merged.dropna(subset=["pm2_5"]).reset_index(drop=True)

# %% [markdown]
# ## 5. Clinical columns: range checks, KNN imputation (numeric), mode imputation (binary)

# %%
NUM_CLIN = ["Age", "BMI", "PhysicalActivity", "PollutionExposure", "DustExposure", "LungFunctionFEV1"]
BIN_CLIN = ["Smoking", "PetAllergy", "FamilyHistoryAsthma", "HistoryOfAllergies", "Wheezing", "ShortnessOfBreath",
            "ChestTightness", "Coughing", "NighttimeSymptoms", "ExerciseInduced"]
ranges = {"Age": (1, 100), "BMI": (10, 60), "PhysicalActivity": (0, 10), "PollutionExposure": (0, 10),
          "DustExposure": (0, 10), "LungFunctionFEV1": (0.5, 6)}
for c, (lo, hi) in ranges.items():
    merged.loc[(merged[c] < lo) | (merged[c] > hi), c] = np.nan
log["clinical_missing_before"] = {c: int(merged[c].isna().sum()) for c in NUM_CLIN + BIN_CLIN}

scaler = StandardScaler()
scaled = scaler.fit_transform(merged[NUM_CLIN])          # NaNs are ignored by StandardScaler
scaled = KNNImputer(n_neighbors=5, weights="distance").fit_transform(scaled)
merged[NUM_CLIN] = scaler.inverse_transform(scaled)
merged[BIN_CLIN] = SimpleImputer(strategy="most_frequent").fit_transform(merged[BIN_CLIN]).astype(int)
for c in ["BMI", "LungFunctionFEV1"]:
    merged[c], n, hi = winsorize(merged[c], 1.5)
log["clinical_missing_after"] = {c: int(merged[c].isna().sum()) for c in NUM_CLIN + BIN_CLIN}
log["rows_clean"] = int(len(merged))

# %% [markdown]
# ## 6. Twelve engineered features
# `add_engineered_features` is copied unchanged into `app.py`, so training and the web app use identical formulas.

# %%
def add_engineered_features(df):
    d = df.copy()
    d["PM25_to_PM10_Ratio"] = (d["pm2_5"] / (d["pm10"] + 1e-6)).clip(0, 1.5)
    d["Rolling_24h_PM25"] = d["pm25_roll24"]
    d["PM25_Lag_1d"] = d["pm25_lag1d"]
    d["Smog_Severity_Index"] = 100 * (0.45 * d["pm2_5"] / 60 + 0.20 * d["pm10"] / 120 + 0.20 * d["no2"] / 80 + 0.15 * d["so2"] / 40)
    d["Temp_Humidity_Discomfort_Index"] = d["temp"] - 0.55 * (1 - d["humidity"] / 100) * (d["temp"] - 14.5)
    d["Atmospheric_Stagnation_Index"] = (d["humidity"] / 100) / (d["wind"] + 1.0)
    n, s = d["no2"] / 80, d["so2"] / 40
    d["NO2_SO2_Synergy_Score"] = n + s + 0.5 * n * s
    d["Diurnal_Temp_Range"] = d["temp_range24"]
    d["Clinical_Vulnerability_Score"] = (0.25 * d["FamilyHistoryAsthma"] + 0.20 * d["Smoking"] + 0.20 * d["HistoryOfAllergies"]
                                         + 0.10 * d["PetAllergy"] + 0.15 * d["DustExposure"] / 10
                                         + 0.10 * ((4.0 - d["LungFunctionFEV1"]).clip(0, 3) / 3))
    d["Pollution_x_Vulnerability"] = d["pm2_5"] * d["Clinical_Vulnerability_Score"]
    d["Pediatric_Geriatric_Risk_Flag"] = ((d["Age"] <= 12) | (d["Age"] >= 60)).astype(int)
    d["Outdoor_Exposure_Risk_Score"] = d["PhysicalActivity"] * d["Smog_Severity_Index"] / 10
    return d


ENGINEERED = ["PM25_to_PM10_Ratio", "Rolling_24h_PM25", "PM25_Lag_1d", "Smog_Severity_Index",
              "Temp_Humidity_Discomfort_Index", "Atmospheric_Stagnation_Index", "NO2_SO2_Synergy_Score",
              "Diurnal_Temp_Range", "Clinical_Vulnerability_Score", "Pollution_x_Vulnerability",
              "Pediatric_Geriatric_Risk_Flag", "Outdoor_Exposure_Risk_Score"]
RAW_FEATURES = ["pm2_5", "pm10", "no2", "so2", "o3", "temp", "humidity", "wind", "Age", "BMI", "Smoking",
                "PhysicalActivity", "PollutionExposure", "DustExposure", "LungFunctionFEV1", "Wheezing",
                "ShortnessOfBreath", "ChestTightness", "Coughing", "NighttimeSymptoms", "ExerciseInduced",
                "PetAllergy", "FamilyHistoryAsthma", "HistoryOfAllergies"]
CANDIDATES = RAW_FEATURES + ENGINEERED

data = add_engineered_features(merged)
assert data[CANDIDATES].isna().sum().sum() == 0, "NaN left in candidate features"
y = data["Asthma_Risk_Level"].astype(int)
X = data[CANDIDATES]
print(f"Clean rows: {len(data):,} | candidate features: {len(CANDIDATES)} ({len(ENGINEERED)} engineered)")

# %% [markdown]
# ## 7. Feature selection (Mutual Information / ANOVA, Random-Forest Gini, RFE)

# %%
# Method 1: SelectKBest with Mutual Information (ANOVA F-test computed alongside)
mi_sel = SelectKBest(partial(mutual_info_classif, random_state=SEED), k=20).fit(X, y)
f_sel = SelectKBest(f_classif, k=20).fit(X, y)
sel = pd.DataFrame({"feature": CANDIDATES, "MI": mi_sel.scores_, "ANOVA_F": f_sel.scores_})

# Method 2: Random-Forest Gini importance + Recursive Feature Elimination
sub = data.sample(8000, random_state=SEED).index
rf_full = RandomForestClassifier(n_estimators=200, class_weight="balanced", max_depth=12, n_jobs=-1, random_state=SEED).fit(X, y)
sel["RF_Gini"] = rf_full.feature_importances_
rfe = RFE(RandomForestClassifier(n_estimators=80, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=SEED),
          n_features_to_select=20, step=2).fit(X.loc[sub], y.loc[sub])
sel["RFE_Rank"] = rfe.ranking_
sel["RFE_Selected"] = rfe.support_

# combine: average of the three ranks (1 = best); keep the top 20
sel["rank_MI"] = sel["MI"].rank(ascending=False)
sel["rank_Gini"] = sel["RF_Gini"].rank(ascending=False)
sel["avg_rank"] = (sel["rank_MI"] + sel["rank_Gini"] + sel["RFE_Rank"]) / 3
sel["is_engineered"] = sel["feature"].isin(ENGINEERED)
sel = sel.sort_values("avg_rank").reset_index(drop=True)
SELECTED = sel["feature"].head(20).tolist()
sel.to_csv(METRICS / "feature_selection_scores.csv", index=False)
print("Selected features:", SELECTED)

# %% [markdown]
# ## 8. Save engineered dataset + selection metadata

# %%
out = data[["record_id", "patient_id", "city", "timestamp"] + CANDIDATES + ["Asthma_Risk_Level", "Asthma_Risk_Label"]]
out.to_csv(PROC / "engineered_features.csv", index=False)
meta = {"candidates": CANDIDATES, "engineered": ENGINEERED, "selected": SELECTED, "class_names": CLASS_NAMES,
        "defaults": {c: float(data[c].median()) for c in CANDIDATES}}
(METRICS / "selected_features.json").write_text(json.dumps(meta, indent=2))
(METRICS / "cleaning_log.json").write_text(json.dumps(log, indent=2))
print(f"Saved engineered_features.csv {out.shape}")

# %% [markdown]
# ## 9. Plots 1-8 with written interpretation

# %%
# Fig 1 - target class distribution
cnt = y.map(dict(enumerate(CLASS_NAMES))).value_counts().reindex(CLASS_NAMES)
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(CLASS_NAMES, cnt.values, color=["#2a9d8f", "#e9c46a", "#f4a261", "#d62828"])
for b, v in zip(bars, cnt.values):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v / len(y):.1%})", ha="center", va="bottom")
ax.set(title="Fig 1 - Asthma risk class distribution", ylabel="Patient-day records", ylim=(0, cnt.max() * 1.2))
save_fig(fig, "fig01_target_class_distribution.png")
interp["fig01"] = (f"The four risk tiers are imbalanced on purpose: Low {cnt['Low'] / len(y):.0%}, Moderate {cnt['Moderate'] / len(y):.0%}, "
                   f"High {cnt['High'] / len(y):.0%}, Critical {cnt['Critical'] / len(y):.0%}. Critical cases are the rarest but the most dangerous, "
                   "so plain accuracy would hide poor performance on them. This is why class weights and recall-focused metrics are used later.")

# Fig 2 - pollutant correlation heatmap
fig, ax = plt.subplots(figsize=(7.5, 6))
corr = timeline[ENV].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, square=True, cbar_kws={"shrink": .8})
ax.set_title("Fig 2 - Pollutant & weather correlation (Open-Meteo timeline)")
save_fig(fig, "fig02_pollutant_correlation_heatmap.png")
interp["fig02"] = (f"PM2.5 and PM10 move together (r = {corr.loc['pm2_5', 'pm10']:.2f}), which is expected because both come from the same combustion, dust and "
                   f"smog sources. PM2.5 also correlates with NO2 (r = {corr.loc['pm2_5', 'no2']:.2f}) and CO (r = {corr.loc['pm2_5', 'co']:.2f}), pointing to traffic and burning. "
                   f"Humidity and wind show weaker links (humidity r = {corr.loc['pm2_5', 'humidity']:.2f}, wind r = {corr.loc['pm2_5', 'wind']:.2f}). "
                   "The strong pollutant-pollutant correlation is a warning about redundancy, which is why feature selection follows.")

# Fig 3 - seasonal / city comparison
india = pd.read_csv(PROC / "india_city_day_standardized.csv", parse_dates=["Date"]).dropna(subset=["PM2.5"])
india["Month"] = india["Date"].dt.month
top = india.groupby("City")["PM2.5"].mean().nlargest(6).index
monthly = india[india.City.isin(top)].groupby(["City", "Month"])["PM2.5"].mean().unstack(0)
monthly.to_csv(METRICS / "india_monthly_pm25.csv")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
monthly.plot(ax=axes[0], marker="o")
axes[0].set(title="South Asian cities: monthly mean PM2.5 (Dataset 2)", xlabel="Month", ylabel="PM2.5 (ug/m3)")
axes[0].set_xticks(range(1, 13))
for city, gg in weekly.groupby("city"):
    axes[1].plot(gg["timestamp"], gg["pm2_5"], marker=".", label=city)
axes[1].set(title="Pakistani cities: weekly mean PM2.5 (Dataset 4)", xlabel="Week", ylabel="PM2.5 (ug/m3)")
axes[1].legend()
axes[1].tick_params(axis="x", rotation=30)
fig.suptitle("Fig 3 - Seasonal and city smog comparison")
save_fig(fig, "fig03_seasonal_city_smog_comparison.png")
peak_m = int(monthly.mean(axis=1).idxmax())
low_m = int(monthly.mean(axis=1).idxmin())
worst_pk = weekly.groupby("city")["pm2_5"].mean().idxmax()
wk_first = weekly.groupby("city")["pm2_5"].apply(lambda s: s.head(4).mean())
wk_last = weekly.groupby("city")["pm2_5"].apply(lambda s: s.tail(4).mean())
trend = "rises" if wk_last.mean() > wk_first.mean() else "falls"
interp["fig03"] = (f"In the multi-year city-day data the six most polluted cities peak in month {peak_m} (mean {monthly.mean(axis=1).max():.0f} ug/m3) and bottom out in month "
                   f"{low_m} ({monthly.mean(axis=1).min():.0f} ug/m3), the winter-smog versus monsoon-wash pattern. In the Pakistani timeline, {worst_pk} has the highest average PM2.5, "
                   f"and the average of the five cities {trend} from {wk_first.mean():.0f} ug/m3 (first 4 weeks) to {wk_last.mean():.0f} ug/m3 (last 4 weeks). "
                   "A three-month window covers only part of the smog season, a bias noted in the report.")

# Fig 4 - PM2.5 vs risk
d4 = data.assign(Risk=data["Asthma_Risk_Label"])
fig, ax = plt.subplots(figsize=(7.5, 4.8))
sns.boxplot(data=d4, x="Risk", y="pm2_5", order=CLASS_NAMES, palette=["#2a9d8f", "#e9c46a", "#f4a261", "#d62828"], showfliers=False, ax=ax)
ax.set(title="Fig 4 - Ambient PM2.5 by asthma risk tier", ylabel="PM2.5 (ug/m3)", xlabel="")
save_fig(fig, "fig04_pm25_vs_asthma_risk_boxplot.png")
med = d4.groupby("Risk")["pm2_5"].median().reindex(CLASS_NAMES)
interp["fig04"] = (f"Median PM2.5 rises from {med['Low']:.0f} ug/m3 (Low) to {med['Moderate']:.0f} (Moderate), {med['High']:.0f} (High) and {med['Critical']:.0f} (Critical). "
                   "The boxes overlap, so PM2.5 alone cannot separate the tiers: a person with a poor clinical profile can reach a high tier on a moderately polluted day. "
                   "That overlap is the reason a model combining pollution with the patient profile is needed instead of a plain AQI reading.")

# Fig 5 - outliers before / after
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
bins = np.linspace(0, max(pm25_before.quantile(0.999), timeline["pm2_5"].max()), 60)
axes[0].hist(pm25_before, bins=bins, color="#e76f51")
axes[0].set(title="Before winsorization", xlabel="PM2.5 (ug/m3)", ylabel="Hourly readings")
axes[1].hist(timeline["pm2_5"], bins=bins, color="#2a9d8f")
axes[1].set(title="After winsorization (3 x IQR fence)", xlabel="PM2.5 (ug/m3)")
fig.suptitle("Fig 5 - PM2.5 outlier handling")
save_fig(fig, "fig05_outlier_before_after.png")
interp["fig05"] = (f"Before capping the maximum hourly PM2.5 was {pm25_before.max():.0f} ug/m3; after capping it is {timeline['pm2_5'].max():.0f} ug/m3 "
                   f"(fence at {capped['pm2_5']['upper_fence']} ug/m3, {capped['pm2_5']['n_capped']} readings changed out of {len(timeline):,}). "
                   "The wide 3 x IQR fence leaves the long right tail of real smog episodes intact and only trims isolated sensor-style spikes.")

# Fig 6 - engineered feature correlations
fig, ax = plt.subplots(figsize=(10, 8.5))
ec = data[ENGINEERED + ["Asthma_Risk_Level"]].corr()
sns.heatmap(ec, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, annot_kws={"size": 7}, cbar_kws={"shrink": .8})
ax.set_title("Fig 6 - Engineered features and risk level correlation")
save_fig(fig, "fig06_engineered_feature_correlations.png")
tc = ec["Asthma_Risk_Level"].drop("Asthma_Risk_Level").abs().sort_values(ascending=False)
interp["fig06"] = (f"The engineered features most correlated with the risk level are {tc.index[0]} (|r| = {tc.iloc[0]:.2f}), {tc.index[1]} ({tc.iloc[1]:.2f}) and {tc.index[2]} ({tc.iloc[2]:.2f}). "
                   f"The weakest is {tc.index[-1]} ({tc.iloc[-1]:.2f}). Several pollution features are strongly inter-correlated (Smog Severity, Rolling 24h, Lag 1d), so they carry overlapping information; "
                   "tree models cope with this, but the SVM and logistic regression benefit from the later feature selection.")

# Fig 7 - Mutual information
s7 = sel.sort_values("MI").tail(20)
fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(s7["feature"], s7["MI"], color=np.where(s7["is_engineered"], "#264653", "#8ab17d"))
ax.set(title="Fig 7 - Mutual Information scores (top 20)", xlabel="Mutual information")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#264653"), plt.Rectangle((0, 0), 1, 1, color="#8ab17d")], labels=["Engineered", "Raw"], loc="lower right")
save_fig(fig, "fig07_mutual_information_scores.png")
n_eng_top = int(s7["is_engineered"].sum())
interp["fig07"] = (f"{n_eng_top} of the top 20 mutual-information features are engineered ones. The top three are {', '.join(sel.sort_values('MI', ascending=False)['feature'].head(3))}. "
                   "Mutual information captures non-linear dependence on the risk tier, so combined indices such as Pollution_x_Vulnerability rank high because they merge environment and patient factors.")

# Fig 8 - RFE ranking + RF Gini
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
s8 = sel.sort_values("RFE_Rank", ascending=False)
axes[0].barh(s8["feature"], s8["RFE_Rank"], color=np.where(s8["RFE_Selected"], "#2a9d8f", "#bbbbbb"))
axes[0].set(title="RFE rank (1 = kept)", xlabel="Rank")
s8b = sel.sort_values("RF_Gini")
axes[1].barh(s8b["feature"], s8b["RF_Gini"], color="#e76f51")
axes[1].set(title="Random Forest Gini importance", xlabel="Importance")
fig.suptitle("Fig 8 - RFE ranking and Random Forest importance")
save_fig(fig, "fig08_rfe_and_rf_importance.png")
agree = len(set(sel[sel.RFE_Selected].feature) & set(sel.sort_values("MI", ascending=False).feature.head(20)))
interp["fig08"] = (f"RFE kept 20 of {len(CANDIDATES)} features, and {agree} of them also appear in the mutual-information top 20, so the two methods largely agree. "
                   f"The Random Forest ranks {sel.sort_values('RF_Gini', ascending=False).feature.iloc[0]} first. Final features are the 20 with the best average rank across MI, Gini and RFE.")

# %%
(PLOTS / "figure_interpretations.json").write_text(json.dumps(interp, indent=2))
for k, v in interp.items():
    print(f"[{k}] {v}\n")
