
import hashlib
import hmac
import json
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(__file__).resolve().parent / "users.db"
MODELS, METRICS, PROC = ROOT / "models", ROOT / "outputs/metrics", ROOT / "data/processed"

CITIES = {"Lahore": (31.5497, 74.3436), "Islamabad": (33.6844, 73.0479), "Rawalpindi": (33.5651, 73.0169),
          "Karachi": (24.8607, 67.0011), "Peshawar": (34.0151, 71.5249)}
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_URL = "https://api.open-meteo.com/v1/forecast"
CLASS_NAMES = ["Low", "Moderate", "High", "Critical"]
RISK_COLORS = {"Low": "#2a9d8f", "Moderate": "#e9c46a", "High": "#f4a261", "Critical": "#d62828"}
ENV_KEYS = ["pm2_5", "pm10", "no2", "so2", "o3", "temp", "humidity", "wind", "roll", "lag", "trange"]
ENV_DEFAULTS = dict(pm2_5=85.0, pm10=150.0, no2=40.0, so2=15.0, o3=45.0, temp=22.0, humidity=60.0, wind=6.0,
                    roll=85.0, lag=80.0, trange=10.0)


# ---------------------------------------------------------------- database & authentication
def db():
    return sqlite3.connect(DB_PATH)


def init_db():
    with db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, full_name TEXT, email TEXT,
            salt TEXT NOT NULL, pw_hash TEXT NOT NULL, created_at TEXT NOT NULL)""")
        con.execute("""CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, city TEXT, model TEXT, risk_level TEXT,
            p_high_critical REAL, pm2_5 REAL, created_at TEXT NOT NULL)""")


def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000)
    return salt, digest.hex()


def register_user(username, full_name, email, password):
    salt, pw_hash = hash_password(password)
    try:
        with db() as con:
            con.execute("INSERT INTO users (username, full_name, email, salt, pw_hash, created_at) VALUES (?,?,?,?,?,?)",
                        (username.strip().lower(), full_name.strip(), email.strip(), salt, pw_hash, datetime.now().isoformat(timespec="seconds")))
        return True, "Account created. You can log in now."
    except sqlite3.IntegrityError:
        return False, "That username is already taken."


def check_login(username, password):
    with db() as con:
        row = con.execute("SELECT id, full_name, salt, pw_hash FROM users WHERE username=?", (username.strip().lower(),)).fetchone()
    if row is None:
        return None
    _, test_hash = hash_password(password, row[2])
    return {"id": row[0], "name": row[1] or username} if hmac.compare_digest(test_hash, row[3]) else None


# ---------------------------------------------------------------- live data (Open-Meteo)
@st.cache_data(ttl=600, show_spinner=False)
def fetch_live(city):
    lat, lon = CITIES[city]
    common = dict(latitude=lat, longitude=lon, past_days=1, forecast_days=2, timezone="Asia/Karachi")
    aq = requests.get(AQ_URL, params={**common, "hourly": "pm10,pm2_5,nitrogen_dioxide,sulphur_dioxide,ozone"}, timeout=20)
    aq.raise_for_status()
    wx = requests.get(WX_URL, params={**common, "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m"}, timeout=20)
    wx.raise_for_status()
    df = pd.DataFrame(aq.json()["hourly"]).merge(pd.DataFrame(wx.json()["hourly"]), on="time")
    df = df.rename(columns={"time": "timestamp", "nitrogen_dioxide": "no2", "sulphur_dioxide": "so2", "ozone": "o3",
                            "temperature_2m": "temp", "relative_humidity_2m": "humidity", "wind_speed_10m": "wind"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    now = pd.Timestamp.now(tz="Asia/Karachi").tz_localize(None).floor("h")
    past = df[(df.timestamp > now - pd.Timedelta(hours=24)) & (df.timestamp <= now)].dropna(subset=["pm2_5"])
    if past.empty:
        raise ValueError("Open-Meteo returned no recent readings for this city.")
    cur = past.iloc[-1]
    lag_row = df[df.timestamp == now - pd.Timedelta(hours=24)]
    env = {"pm2_5": cur.pm2_5, "pm10": cur.pm10, "no2": cur.no2, "so2": cur.so2, "o3": cur.o3, "temp": cur.temp,
           "humidity": cur.humidity, "wind": cur.wind, "roll": past.pm2_5.mean(),
           "lag": float(lag_row.pm2_5.iloc[0]) if len(lag_row) and pd.notna(lag_row.pm2_5.iloc[0]) else past.pm2_5.iloc[0],
           "trange": past.temp.max() - past.temp.min()}
    env = {k: float(0 if pd.isna(v) else v) for k, v in env.items()}
    future = df[(df.timestamp > now) & (df.timestamp <= now + pd.Timedelta(hours=24))]
    return {"env": env, "past": past, "future": future, "asof": cur.timestamp}


def pm25_to_aqi(pm):
    """US AQI from PM2.5 (EPA 2024 breakpoints)."""
    bp = [(0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 125.4, 151, 200),
          (125.5, 225.4, 201, 300), (225.5, 325.4, 301, 500)]
    for lo, hi, a, b in bp:
        if pm <= hi:
            return round((b - a) / (hi - lo) * (max(pm, lo) - lo) + a)
    return 500


# ---------------------------------------------------------------- features (same formulas as script 02)
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


# ---------------------------------------------------------------- models & prediction
@st.cache_resource(show_spinner=False)
def load_models():
    meta_path = MODELS / "model_meta.json"
    if not meta_path.exists():
        return None, {}
    meta = json.loads(meta_path.read_text())
    loaded = {}
    for name, fname in meta["model_files"].items():
        if (MODELS / fname).exists():
            loaded[name] = joblib.load(MODELS / fname)
    return meta, loaded


def build_input_row(env, patient):
    row = {"pm2_5": env["pm2_5"], "pm10": env["pm10"], "no2": env["no2"], "so2": env["so2"], "o3": env["o3"],
           "temp": env["temp"], "humidity": env["humidity"], "wind": env["wind"], "pm25_roll24": env["roll"],
           "pm25_lag1d": env["lag"], "temp_range24": env["trange"], **patient}
    return add_engineered_features(pd.DataFrame([row]))


def predict_risk(model, meta, env, patient):
    X = build_input_row(env, patient)[meta["features"]]
    proba = model.predict_proba(X)[0]
    p_hi = float(proba[2:].sum())
    level = int(proba.argmax())
    escalated = False
    if p_hi >= meta["alert_threshold"] and level < 2:     # safety rule: never under-warn a likely High/Critical case
        level, escalated = 2, True
    return CLASS_NAMES[level], proba, p_hi, escalated


# ---------------------------------------------------------------- recommendation engine
def recommend(level, env, patient, future=None):
    pm, age = env["pm2_5"], patient["Age"]
    n_sym = sum(patient[k] for k in ["Wheezing", "ShortnessOfBreath", "ChestTightness", "Coughing", "NighttimeSymptoms"])
    child = age <= 12
    recs = []
    mask = {"Low": "No mask needed for short outdoor trips. Keep one in your bag if you are sensitive to dust.",
            "Moderate": "Wear a well-fitted N95/KN95 mask if you will be outside for more than 30 minutes or near traffic.",
            "High": "Wear a well-fitted N95/KN95 mask for any time outdoors, including commuting.",
            "Critical": "Avoid going outside. If you must, use an N99/P100-class respirator and keep the trip short."}[level]
    if child and level != "Low":
        mask += " Adult masks rarely seal on small faces, so for children it is safer to stay indoors than to rely on a mask."
    recs.append(("Mask", mask))
    purifier = {"Low": "Normal ventilation is fine. A purifier is optional.",
                "Moderate": "Run a HEPA purifier on medium in the bedroom or main living room, with windows closed during peak hours.",
                "High": "Run the HEPA purifier on high/auto in the room where you spend most time, keep windows and doors shut, and avoid frying or burning incense indoors.",
                "Critical": "Run the purifier on its highest setting in one sealed 'clean room', seal door gaps with a towel, and do not vacuum or sweep dry."}[level]
    recs.append(("Indoor air purifier", purifier + " Check the filter - a clogged HEPA filter does little."))
    inhaler = {"Low": "Keep your reliever inhaler where you usually do and check the dose counter.",
               "Moderate": "Keep your reliever inhaler within reach today, take preventer medication exactly as prescribed, and check it is not empty or expired.",
               "High": "Carry your reliever inhaler (with spacer if you use one) at all times and review your written asthma action plan. Tell someone nearby that today is a higher-risk day.",
               "Critical": "Keep the reliever on your person, review your asthma action plan now, and make sure a family member knows where your medication is."}[level]
    if patient["ExerciseInduced"] and level != "Low":
        inhaler += " You have exercise-induced symptoms, so skip outdoor exercise today."
    recs.append(("Rescue inhaler readiness", inhaler))
    best = None
    if future is not None and len(future) >= 3:
        r3 = future["pm2_5"].rolling(3).mean()
        if r3.notna().any():
            end = r3.idxmin()
            start_ts = future.loc[end - 2, "timestamp"] if (end - 2) in future.index else future.loc[end, "timestamp"]
            best = f"Open-Meteo's forecast shows the cleanest 3-hour window in the next 24 hours is {start_ts:%a %H:%M}-{future.loc[end, 'timestamp'] + pd.Timedelta(hours=1):%H:%M} (PM2.5 about {r3.min():.0f} ug/m3)."
    hours = {"Low": "Any time is fine.",
             "Moderate": "Late morning to mid-afternoon (about 12:00-16:00) is usually cleaner than early morning and night, when smog is trapped near the ground.",
             "High": "Go outside only if necessary and keep it brief, ideally 12:00-16:00. Avoid early morning and after sunset.",
             "Critical": "Stay indoors. Postpone outdoor errands and exercise until air quality improves."}[level]
    recs.append(("Safer outdoor hours", hours + (" " + best if best else "")))
    alert = None
    if level == "Critical" or (level == "High" and (n_sym >= 3 or patient["LungFunctionFEV1"] < 1.8)) or (level == "High" and patient["NighttimeSymptoms"]):
        alert = ("Contact your doctor or pulmonologist today to review your treatment. Seek emergency care immediately if you cannot speak a full sentence, "
                 "your lips or fingertips look blue, or your reliever inhaler is not helping (in Pakistan, e.g. Rescue 1122 or Edhi 115).")
    elif level == "High":
        alert = "Book a check-up soon if symptoms are becoming more frequent. Call your doctor if you need your reliever more than usual."
    elif level == "Moderate" and n_sym >= 3:
        alert = "You report several symptoms. Mention them at your next visit, or sooner if they get worse."
    if alert:
        recs.append(("Doctor consultation", alert))
    extra = []
    if patient["Smoking"]:
        extra.append("Smoking raises your sensitivity to smog; avoid smoking on high-pollution days.")
    if env["o3"] > 100:
        extra.append("Ozone is high - afternoon outdoor activity may irritate airways today.")
    if env["humidity"] > 85 and env["wind"] < 5:
        extra.append("Humid, almost still air is trapping pollutants near the ground.")
    if extra:
        recs.append(("Other notes", " ".join(extra)))
    return recs


# ---------------------------------------------------------------- UI pieces
def gauge(p_hi, threshold):
    steps = [{"range": [0, threshold * 50], "color": "#2a9d8f"}, {"range": [threshold * 50, threshold * 100], "color": "#e9c46a"},
             {"range": [threshold * 100, min(threshold * 100 + 30, 100)], "color": "#f4a261"},
             {"range": [min(threshold * 100 + 30, 100), 100], "color": "#d62828"}]
    fig = go.Figure(go.Indicator(mode="gauge+number", value=p_hi * 100, number={"suffix": "%"},
                                 title={"text": "Probability of High or Critical risk"},
                                 gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#222"}, "steps": steps,
                                        "threshold": {"line": {"color": "black", "width": 4}, "value": threshold * 100}}))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def reset_env_defaults():
    for k in ENV_KEYS:
        st.session_state.setdefault(f"env_{k}", ENV_DEFAULTS[k])


def apply_live(city):
    try:
        data = fetch_live(city)
        for k, v in data["env"].items():
            st.session_state[f"env_{k}"] = round(float(v), 2)
        st.session_state["live"] = {"city": city, **data}
        st.session_state["live_error"] = None
    except Exception as exc:
        st.session_state["live_error"] = f"Could not reach Open-Meteo ({type(exc).__name__}). Enter values manually below."


def page_predictor(user):
    meta, models = load_models()
    st.header("Asthma & Smog Risk Predictor")
    if meta is None or not models:
        st.error("Trained models not found. Run scripts 01 -> 02 -> 03 first (see README).")
        return
    reset_env_defaults()
    st.subheader("1. Environment")
    c1, c2 = st.columns([2, 1])
    city = c1.selectbox("City", list(CITIES))
    c2.write("")
    c2.button("Fetch live data", on_click=apply_live, args=(city,), use_container_width=True)
    if st.session_state.get("live_error"):
        st.warning(st.session_state["live_error"])
    live = st.session_state.get("live")
    if live and live["city"] == city:
        aqi = pm25_to_aqi(live["env"]["pm2_5"])
        st.success(f"Live data for {city} as of {live['asof']:%d %b %Y %H:%M} (Pakistan time). Approx. US AQI from PM2.5: {aqi}.")
    e = {}
    cols = st.columns(4)
    spec = [("pm2_5", "PM2.5 (ug/m3)", 0.0, 800.0), ("pm10", "PM10 (ug/m3)", 0.0, 1200.0), ("no2", "NO2 (ug/m3)", 0.0, 400.0),
            ("so2", "SO2 (ug/m3)", 0.0, 300.0), ("o3", "Ozone (ug/m3)", 0.0, 400.0), ("temp", "Temperature (C)", -5.0, 50.0),
            ("humidity", "Humidity (%)", 0.0, 100.0), ("wind", "Wind (km/h)", 0.0, 80.0)]
    for i, (k, label, lo, hi) in enumerate(spec):
        e[k] = cols[i % 4].number_input(label, lo, hi, key=f"env_{k}", step=1.0)
    with st.expander("Time-based inputs (filled automatically from live data)"):
        t1, t2, t3 = st.columns(3)
        e["roll"] = t1.number_input("Average PM2.5, last 24 h", 0.0, 800.0, key="env_roll")
        e["lag"] = t2.number_input("PM2.5 one day ago", 0.0, 800.0, key="env_lag")
        e["trange"] = t3.number_input("Temperature range, last 24 h (C)", 0.0, 30.0, key="env_trange")

    st.subheader("2. Patient profile")
    with st.form("patient"):
        a, b, c = st.columns(3)
        age = a.number_input("Age", 1, 100, 35)
        bmi = b.number_input("BMI", 10.0, 60.0, 25.0)
        fev_known = c.checkbox("I know my FEV1 (litres)")
        fev = c.number_input("FEV1 (L)", 0.5, 6.0, 2.8) if fev_known else float(meta["defaults"].get("LungFunctionFEV1", 2.8))
        a, b, c = st.columns(3)
        smoking = a.checkbox("Smoker")
        fam = b.checkbox("Family history of asthma")
        allergy = c.checkbox("History of allergies")
        pet = a.checkbox("Pet allergy")
        exercise = b.checkbox("Symptoms triggered by exercise")
        night = c.checkbox("Night-time symptoms")
        a, b, c = st.columns(3)
        activity = a.slider("Outdoor physical activity today (0-10)", 0.0, 10.0, 4.0)
        pol_exp = b.slider("Daily pollution exposure (0-10)", 0.0, 10.0, 5.0)
        dust = c.slider("Dust exposure at home/work (0-10)", 0.0, 10.0, 4.0)
        st.write("Current symptoms")
        s1, s2, s3, s4 = st.columns(4)
        wheeze, sob, chest, cough = s1.checkbox("Wheezing"), s2.checkbox("Shortness of breath"), s3.checkbox("Chest tightness"), s4.checkbox("Coughing")
        names = list(models)
        default_idx = names.index(meta["best_model"]) if meta["best_model"] in names else 0
        model_name = st.selectbox("Model", names, index=default_idx)
        go_btn = st.form_submit_button("Predict my risk", type="primary")
    if not go_btn:
        return
    patient = {"Age": age, "BMI": bmi, "Smoking": int(smoking), "PhysicalActivity": activity, "PollutionExposure": pol_exp,
               "DustExposure": dust, "LungFunctionFEV1": fev, "Wheezing": int(wheeze), "ShortnessOfBreath": int(sob),
               "ChestTightness": int(chest), "Coughing": int(cough), "NighttimeSymptoms": int(night),
               "ExerciseInduced": int(exercise), "PetAllergy": int(pet), "FamilyHistoryAsthma": int(fam),
               "HistoryOfAllergies": int(allergy)}
    level, proba, p_hi, escalated = predict_risk(models[model_name], meta, e, patient)
    with db() as con:
        con.execute("INSERT INTO assessments (user_id, city, model, risk_level, p_high_critical, pm2_5, created_at) VALUES (?,?,?,?,?,?,?)",
                    (user["id"], city, model_name, level, p_hi, e["pm2_5"], datetime.now().isoformat(timespec="seconds")))
    st.subheader("3. Result")
    st.markdown(f"<div style='padding:14px;border-radius:10px;background:{RISK_COLORS[level]};color:{'#222' if level == 'Moderate' else 'white'};"
                f"font-size:1.4rem;font-weight:700;text-align:center'>Asthma attack risk: {level.upper()}</div>", unsafe_allow_html=True)
    if escalated:
        st.caption(f"Raised to High by the safety rule: P(High or Critical) = {p_hi:.0%} is above the alert threshold of {meta['alert_threshold']:.0%}.")
    g1, g2 = st.columns(2)
    g1.plotly_chart(gauge(p_hi, meta["alert_threshold"]), use_container_width=True)
    pf = pd.DataFrame({"Risk tier": CLASS_NAMES, "Probability": proba})
    g2.plotly_chart(px.bar(pf, x="Risk tier", y="Probability", color="Risk tier", color_discrete_map=RISK_COLORS, range_y=[0, 1]).update_layout(
        height=280, showlegend=False, margin=dict(l=10, r=10, t=30, b=10)), use_container_width=True)
    st.subheader("4. Personalised recommendations")
    future = live["future"].reset_index(drop=True) if live and live["city"] == city else None
    for title, text in recommend(level, e, patient, future):
        st.markdown(f"**{title}.** {text}")
    st.caption("BreatheSafe is a decision-support tool built for a university lab. It is not a medical device and does not replace your doctor's advice.")


def page_dashboard():
    st.header("Analytical Dashboard")
    meta, _ = load_models()
    t1, t2, t3, t4 = st.tabs(["City smog trends", "Dataset insights", "Model comparison", "Feature importance"])
    with t1:
        f = METRICS / "city_pm25_weekly.csv"
        if f.exists():
            w = pd.read_csv(f, parse_dates=["timestamp"])
            st.plotly_chart(px.line(w, x="timestamp", y="pm2_5", color="city", markers=True, labels={"pm2_5": "PM2.5 (ug/m3)", "timestamp": "Week"},
                                    title="Weekly mean PM2.5 by city (training window)"), use_container_width=True)
        f = METRICS / "india_monthly_pm25.csv"
        if f.exists():
            m = pd.read_csv(f, index_col=0)
            st.plotly_chart(px.imshow(m.T, aspect="auto", color_continuous_scale="YlOrRd", labels=dict(x="Month", y="City", color="PM2.5"),
                                      title="Seasonal PM2.5 profile of South Asian cities (Kaggle city-day data)"), use_container_width=True)
        if not (METRICS / "city_pm25_weekly.csv").exists():
            st.info("Run scripts 01 and 02 to generate the dashboard data.")
    with t2:
        f = PROC / "engineered_features.csv"
        if f.exists():
            d = pd.read_csv(f)
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.pie(d, names="Asthma_Risk_Label", color="Asthma_Risk_Label", color_discrete_map=RISK_COLORS,
                                   category_orders={"Asthma_Risk_Label": CLASS_NAMES}, title="Risk tier distribution"), use_container_width=True)
            c2.plotly_chart(px.box(d.sample(min(5000, len(d)), random_state=1), x="Asthma_Risk_Label", y="pm2_5", color="Asthma_Risk_Label",
                                   color_discrete_map=RISK_COLORS, category_orders={"Asthma_Risk_Label": CLASS_NAMES}, points=False,
                                   title="PM2.5 by risk tier").update_layout(showlegend=False), use_container_width=True)
            st.plotly_chart(px.histogram(d, x="Age", color="Asthma_Risk_Label", color_discrete_map=RISK_COLORS, nbins=30, barmode="stack",
                                         category_orders={"Asthma_Risk_Label": CLASS_NAMES}, title="Age profile by risk tier"), use_container_width=True)
            st.caption(f"{len(d):,} patient-day records after cleaning.")
        else:
            st.info("Run scripts 01 and 02 first.")
    with t3:
        if meta:
            m = pd.DataFrame(meta["metrics"])
            show = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "HighRisk_Recall", "Train_Time_s", "Inference_ms_per_pred"]
            st.dataframe(m[show], use_container_width=True, hide_index=True)
            long = m.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"], var_name="Metric")
            st.plotly_chart(px.bar(long, x="Metric", y="value", color="Model", barmode="group", range_y=[0, 1], title="Test-set metrics"), use_container_width=True)
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.bar(m, x="Model", y="Train_Time_s", title="Training time (s)"), use_container_width=True)
            c2.plotly_chart(px.bar(m, x="Model", y="Inference_ms_per_pred", title="Inference time (ms per prediction)"), use_container_width=True)
            st.caption(f"Best model on validation macro-F1: {meta['best_model']}. High-risk alert threshold: {meta['alert_threshold']:.0%}.")
        else:
            st.info("Run script 03 first.")
    with t4:
        f = METRICS / "feature_selection_scores.csv"
        if f.exists():
            s = pd.read_csv(f)
            st.plotly_chart(px.bar(s.sort_values("RF_Gini").tail(20), x="RF_Gini", y="feature", orientation="h", color="is_engineered",
                                   title="Random Forest importance (top 20)", height=600), use_container_width=True)
            st.plotly_chart(px.bar(s.sort_values("MI").tail(20), x="MI", y="feature", orientation="h", color="is_engineered",
                                   title="Mutual information (top 20)", height=600), use_container_width=True)
        else:
            st.info("Run script 02 first.")




def page_history(user):
    st.header("My assessment history")
    with db() as con:
        h = pd.read_sql_query(
            "SELECT created_at AS time, city, model, risk_level AS risk, ROUND(p_high_critical*100,1) AS 'P(High/Critical) %', "
            "pm2_5 AS 'PM2.5' FROM assessments WHERE user_id=? ORDER BY id DESC", 
            con, 
            params=(user["id"],)
        )
    
    if len(h) > 0:
        st.dataframe(h, use_container_width=True, hide_index=True)
    else:
        st.info("No assessments yet.")


def page_about():
    st.header("About BreatheSafe")
    st.write("BreatheSafe combines live smog and weather data from Open-Meteo with a patient's health profile to estimate asthma attack risk "
             "(Low / Moderate / High / Critical).")
    st.write(" Three models were trained: Random Forest, Gradient Boosting and SVM, with Logistic Regression as a baseline.")
    st.write("**Group Members:** Abdul Hannan Khan (Group Leader), Muhammad Ali Khan , Hafsa Qureshi  " )
    st.write("Machine Learning Lab -Fall 2026")   
    st.write("Department of Computer Science- NUML")
    st.warning("The risk label used to train the models is a documented proxy built from epidemiological relationships, not a recorded clinical outcome. "
               "Treat results as decision support, not diagnosis.")



import random
import re

def page_edit_profile(user):
    st.header("Edit Profile & Password")
    st.write("Update Your Profile Details Here.")
    
    # Database se current user ki latest details fetch karo
    with db() as con:
        row = con.execute("SELECT username, full_name, email, salt, pw_hash FROM users WHERE id=?", (user["id"],)).fetchone()
        
    if not row:
        st.error("User not found.")
        return
        
    curr_username, curr_name, curr_email, db_salt, db_hash = row
    
    # Session state initialization for profile email change verification
    if "profile_edit_step" not in st.session_state:
        st.session_state["profile_edit_step"] = 1
        
    # --- SECTION 1: UPDATE PROFILE DETAILS ---
    st.subheader("1. Update Personal Information")
    
    if st.session_state["profile_edit_step"] == 1:
        with st.form("edit_profile_form"):
            new_name = st.text_input("Full Name", value=curr_name or "")
            new_email = st.text_input("Email", value=curr_email or "")
            new_username = st.text_input("Username", value=curr_username or "")
            
            submit_profile = st.form_submit_button("Save Profile Changes", type="primary")
            
            if submit_profile:
                email_regex = r"^[\w\.-]+@[\w\.-]+\.\w+$"
                
                if not new_name.strip():
                    st.error("Full name cannot be empty.")
                elif not re.match(email_regex, new_email.strip()):
                    st.error("Please enter a valid email address.")
                elif len(new_username.strip()) < 3:
                    st.error("Username must be at least 3 characters.")
                else:
                    try:
                        with db() as con:
                            # Check karo ke naya username kisi aur ke pas toh nahi hai
                            existing = con.execute("SELECT id FROM users WHERE username=? AND id!=?", 
                                                  (new_username.strip().lower(), user["id"])).fetchone()
                        if existing:
                            st.error("Username is not free.")
                        else:
                            # Agar email change ki hai, toh OTP verification trigger karo
                            if new_email.strip().lower() != curr_email.strip().lower():
                                profile_otp = str(random.randint(100000, 999999))
                                
                                # Real email send function call hoga
                                sent, msg = send_verification_email(new_email.strip(), profile_otp, purpose="Email Update")
                                
                                if sent:
                                    st.session_state["temp_profile_data"] = {
                                        "name": new_name.strip(),
                                        "email": new_email.strip(),
                                        "username": new_username.strip().lower()
                                    }
                                    st.session_state["profile_otp"] = profile_otp
                                    st.session_state["profile_edit_step"] = 2
                                    st.success("Verification OTP has been sent to your new email address!")
                                    st.rerun()
                                else:
                                    st.error(f"Could not send verification email: {msg}")
                            else:
                                # Agar email change nahi ki, toh direct update kar do
                                with db() as con:
                                    con.execute("UPDATE users SET full_name=?, username=? WHERE id=?",
                                                (new_name.strip(), new_username.strip().lower(), user["id"]))
                                
                                st.session_state["user"]["name"] = new_name.strip()
                                st.success("Profile successfully updated!")
                                st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                        
    elif st.session_state["profile_edit_step"] == 2:
        # Screen par ab koi demo OTP show nahi hoga, strictly email check karni paregi!
        
        with st.form("verify_profile_email_form"):
            st.info(f"A verification code has been sent to your new email: **{st.session_state['temp_profile_data']['email']}**. Please check your inbox.")
            entered_p_otp = st.text_input("Enter 6-digit Verification Code")
            
            c1, c2 = st.columns(2)
            verify_p_btn = c1.form_submit_button("Verify & Update Email", type="primary")
            cancel_p_btn = c2.form_submit_button("Cancel")
            
            if cancel_p_btn:
                st.session_state["profile_edit_step"] = 1
                st.rerun()
                
            if verify_p_btn:
                if entered_p_otp.strip() != st.session_state.get("profile_otp"):
                    st.error("Invalid verification code. Please check your email and try again.")
                else:
                    data = st.session_state["temp_profile_data"]
                    with db() as con:
                        con.execute("UPDATE users SET full_name=?, email=?, username=? WHERE id=?",
                                    (data["name"], data["email"], data["username"], user["id"]))
                    
                    st.session_state["user"]["name"] = data["name"]
                    st.success("Email verified and profile updated successfully!")
                    st.session_state["profile_edit_step"] = 1
                    st.session_state.pop("temp_profile_data", None)
                    st.session_state.pop("profile_otp", None)
                    st.rerun()

    st.markdown("---")

    # --- SECTION 2: CHANGE PASSWORD ---
    st.subheader("2. Change Password")
    with st.form("change_password_form"):
        current_pass = st.text_input("Current Password", type="password")
        new_pass = st.text_input("New Password (8+ characters)", type="password")
        confirm_pass = st.text_input("Confirm New Password", type="password")
        
        submit_pass = st.form_submit_button("Update Password", type="primary")
        
        if submit_pass:
            _, test_hash = hash_password(current_pass, db_salt)
            
            if not hmac.compare_digest(test_hash, db_hash):
                st.error("Current password is wrong.")
            elif len(new_pass) < 8:
                st.error("Enter 8+ Character.")
            elif new_pass != confirm_pass:
                st.error("New Password Not Match.")
            else:
                new_salt, new_pw_hash = hash_password(new_pass)
                with db() as con:
                    con.execute("UPDATE users SET salt=?, pw_hash=? WHERE id=?", 
                                (new_salt, new_pw_hash, user["id"]))
                st.success("Password Change Successfully")
# def page_edit_profile(user):
#     st.header("Edit Profile & Password")
#     st.write("Update Your Profile Details Here .")
    
#     # Database se current user ki latest details fetch karo
#     with db() as con:
#         row = con.execute("SELECT username, full_name, email, salt, pw_hash FROM users WHERE id=?", (user["id"],)).fetchone()
        
#     if not row:
#         st.error("User not found.")
#         return
        
#     curr_username, curr_name, curr_email, db_salt, db_hash = row
    
#     # --- SECTION 1: UPDATE PROFILE DETAILS ---
#     st.subheader("1. Update Personal Information")
#     with st.form("edit_profile_form"):
#         new_name = st.text_input("Full Name", value=curr_name or "")
#         new_email = st.text_input("Email", value=curr_email or "")
#         new_username = st.text_input("Username", value=curr_username or "")
        
#         submit_profile = st.form_submit_button("Save Profile Changes", type="primary")
        
#         if submit_profile:
#             email_regex = r"^[\w\.-]+@[\w\.-]+\.\w+$"
            
#             if not new_name.strip():
#                 st.error("Full name cannot be empty.")
#             elif not re.match(email_regex, new_email.strip()):
#                 st.error("Please enter a valid email address.")
#             elif len(new_username.strip()) < 3:
#                 st.error("Username must be at least 3 characters.")
#             else:
#                 try:
#                     with db() as con:
#                         # Check karo ke naya username kisi aur ke pas toh nahi hai
#                         existing = con.execute("SELECT id FROM users WHERE username=? AND id!=?", 
#                                               (new_username.strip().lower(), user["id"])).fetchone()
#                         if existing:
#                             st.error("Username is not free.")
#                         else:
#                             con.execute("UPDATE users SET full_name=?, email=?, username=? WHERE id=?",
#                                         (new_name.strip(), new_email.strip(), new_username.strip().lower(), user["id"]))
                            
#                             # Session state mein name update karo
#                             st.session_state["user"]["name"] = new_name.strip()
#                             st.success("Profile successfully updated!")
#                             st.rerun()
#                 except Exception as e:
#                     st.error(f"Error: {e}")

#     st.markdown("---")

#     # --- SECTION 2: CHANGE PASSWORD ---
#     st.subheader("2. Change Password")
#     with st.form("change_password_form"):
#         current_pass = st.text_input("Current Password", type="password")
#         new_pass = st.text_input("New Password (8+ characters)", type="password")
#         confirm_pass = st.text_input("Confirm New Password", type="password")
        
#         submit_pass = st.form_submit_button("Update Password" , type="primary")
        
#         if submit_pass:
#             # Pehle check karo ke current password theek hai ya nahi
#             _, test_hash = hash_password(current_pass, db_salt)
            
#             if not hmac.compare_digest(test_hash, db_hash):
#                 st.error("Current password  is wrong .")
#             elif len(new_pass) < 8:
#                 st.error("Enter 8+ Character ")
#             elif new_pass != confirm_pass:
#                 st.error("New Password Not Match.")
#             else:
#                 # Naya salt aur hash generate karke update karo
#                 new_salt, new_pw_hash = hash_password(new_pass)
#                 with db() as con:
#                     con.execute("UPDATE users SET salt=?, pw_hash=? WHERE id=?", 
#                                 (new_salt, new_pw_hash, user["id"]))
#                 st.success("Password Change Successfully ")                      


# import random
# import re

# def auth_screen():
#     st.title("BreatheSafe")
#     st.caption("Smog & Asthma Risk Predictor")
    
#     # 3 Tabs: Log in, Sign up, Forgot Password
#     tab_login, tab_signup, tab_forgot = st.tabs(["Log in", "Sign up", "Forgot Password"])
    
#     # ---------------- 1. LOGIN TAB ----------------
#     with tab_login:
#         with st.form("login"):
#             u, p = st.text_input("Username"), st.text_input("Password", type="password")
#             if st.form_submit_button("Log in", type="primary"):
#                 user = check_login(u, p)
#                 if user:
#                     st.session_state["user"] = user
#                     st.rerun()
#                 else:
#                     st.error("Wrong username or password.")
                    
#     # ---------------- 2. SIGN UP TAB (WITH VALIDATION & OTP) ----------------
#     with tab_signup:
#         st.subheader("Create a new account")
        
#         if "signup_step" not in st.session_state:
#             st.session_state["signup_step"] = 1  # Step 1: Form Inputs, Step 2: OTP Verification
            
#         if st.session_state["signup_step"] == 1:
#             with st.form("signup_step1"):
#                 n = st.text_input("Full name")
#                 em = st.text_input("Email (e.g., user@example.com)")
#                 u = st.text_input("Choose a username (3+ chars)")
#                 p = st.text_input("Password (8+ characters)", type="password")
#                 p2 = st.text_input("Confirm password", type="password")
                
#                 submit_signup = st.form_submit_button("Verify Email & Send OTP", type="primary")
                
#                 if submit_signup:
#                     # Email validation regex pattern
#                     email_regex = r"^[\w\.-]+@[\w\.-]+\.\w+$"
                    
#                     if not n.strip():
#                         st.error("Full name cannot be empty.")
#                     elif not re.match(email_regex, em.strip()):
#                         st.error("Please enter a valid email address (must contain '@' and domain).")
#                     elif len(u.strip()) < 3:
#                         st.error("Username needs at least 3 characters.")
#                     elif len(p) < 8:
#                         st.error("Password must be at least 8 characters.")
#                     elif p != p2:
#                         st.error("Passwords do not match.")
#                     else:
#                         # Check if username already exists in database
#                         with db() as con:
#                             exists = con.execute("SELECT id FROM users WHERE username=?", (u.strip().lower(),)).fetchone()
#                         if exists:
#                             st.error("That username is already taken. Choose another one.")
#                         else:
#                             # Temp data save karo session state mein aur OTP generate karo
#                             st.session_state["temp_signup"] = {
#                                 "name": n.strip(),
#                                 "email": em.strip(),
#                                 "username": u.strip().lower(),
#                                 "password": p
#                             }
#                             signup_otp = str(random.randint(100000, 999999))
#                             st.session_state["signup_otp"] = signup_otp
#                             st.session_state["signup_step"] = 2
#                             st.success(f"Validation successful! (Demo Mode - Verification OTP is: **{signup_otp}**)")
#                             st.rerun()
                            
#         elif st.session_state["signup_step"] == 2:
#             with st.form("signup_step2"):
#                 st.info(f"A verification code has been sent for **{st.session_state['temp_signup']['email']}**. Enter it below to complete signup.")
#                 entered_signup_otp = st.text_input("Enter 6-digit Verification Code")
                
#                 c1, c2 = st.columns(2)
#                 verify_signup_btn = c1.form_submit_button("Complete Registration", type="primary")
#                 back_signup_btn = c2.form_submit_button("Back / Edit Info")
                
#                 if back_signup_btn:
#                     st.session_state["signup_step"] = 1
#                     st.rerun()
                    
#                 if verify_signup_btn:
#                     if entered_signup_otp.strip() != st.session_state.get("signup_otp"):
#                         st.error("Invalid verification code. Please check and try again.")
#                     else:
#                         data = st.session_state["temp_signup"]
#                         ok, msg = register_user(data["username"], data["name"], data["email"], data["password"])
#                         if ok:
#                             st.success("Account successfully verified and created! You can now log in.")
#                             # Clear signup states
#                             st.session_state["signup_step"] = 1
#                             st.session_state.pop("temp_signup", None)
#                             st.session_state.pop("signup_otp", None)
#                         else:
#                             st.error(msg)

#     # ---------------- 3. FORGOT PASSWORD TAB (WITH OTP) ----------------
#     with tab_forgot:
#         st.subheader("Reset Password via OTP")
        
#         if "otp_step" not in st.session_state:
#             st.session_state["otp_step"] = 1
        
#         if st.session_state["otp_step"] == 1:
#             with st.form("forgot_step1"):
#                 f_user = st.text_input("Enter your Username or Email", key="f_user_input")
#                 send_otp_btn = st.form_submit_button("Send OTP")
                
#                 if send_otp_btn:
#                     with db() as con:
#                         row = con.execute("SELECT id, username, email FROM users WHERE username=? OR email=?", 
#                                           (f_user.strip().lower(), f_user.strip())).fetchone()
#                     if row:
#                         generated_otp = str(random.randint(100000, 999999))
#                         st.session_state["reset_user_id"] = row[0]
#                         st.session_state["saved_otp"] = generated_otp
#                         st.session_state["otp_step"] = 2
#                         st.success(f"OTP generated successfully! (Demo Mode - Your OTP code is: **{generated_otp}**)")
#                         st.rerun()
#                     else:
#                         st.error("No account found with this username or email.")
                        
#         elif st.session_state["otp_step"] == 2:
#             st.warning(f"🔑 **Demo OTP Code:** `{st.session_state.get('saved_otp')}` (Yeh code yahan copy/paste kar dein)") 

#             with st.form("forgot_step2"):
#                 st.info("An OTP code has been generated. Enter it below along with your new password.")
#                 entered_otp = st.text_input("Enter 6-digit OTP")
#                 new_pass = st.text_input("New Password (8+ characters)", type="password")
#                 confirm_pass = st.text_input("Confirm New Password", type="password")
                
#                 c1, c2 = st.columns(2)
#                 verify_btn = c1.form_submit_button("Reset Password", type="primary")
#                 back_btn = c2.form_submit_button("Cancel / Back")
                
#                 if back_btn:
#                     st.session_state["otp_step"] = 1
#                     st.rerun()
                    
#                 if verify_btn:
#                     if entered_otp.strip() != st.session_state.get("saved_otp"):
#                         st.error("Invalid OTP code. Please check and try again.")
#                     elif len(new_pass) < 8:
#                         st.error("Password must be at least 8 characters.")
#                     elif new_pass != confirm_pass:
#                         st.error("Passwords do not match.")
#                     else:
#                         salt, pw_hash = hash_password(new_pass)
#                         with db() as con:
#                             con.execute("UPDATE users SET salt=?, pw_hash=? WHERE id=?", 
#                                         (salt, pw_hash, st.session_state["reset_user_id"]))
                        
#                         st.success("Password successfully reset! You can now log in using your new password.")
#                         st.session_state["otp_step"] = 1
#                         st.session_state.pop("saved_otp", None)
#                         st.session_state.pop("reset_user_id", None)

import random
import re
import smtplib
from email.message import EmailMessage

# --- EMAIL SENDING HELPER FUNCTION ---
def send_verification_email(to_email, otp_code, purpose="Verification"):
    
    SENDER_EMAIL = "helpinstudies12@gmail.com"
    APP_PASSWORD = "blug onrx xqtk fxrl" 
    
    msg = EmailMessage()
    msg["Subject"] = f"BreatheSafe - {purpose} OTP Code"
    msg["From"] = f"BreatheSafe App <{SENDER_EMAIL}>"
    msg["To"] = to_email
    msg.set_content(
        f"Assalam-o-Alaikum,\n\n"
        f"Your BreatheSafe {purpose} OTP code is: {otp_code}\n\n"
        f"Please enter this 6-digit code in the app to verify your email address.\n"
        f"If you did not request this, please ignore this email.\n\n"
        f"Regards,\nBreatheSafe Team (NUML)"
    )
    
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)
        return True, "OTP sent successfully to your email!"
    except Exception as e:
        return False, f"Failed to send email: {e}"


def auth_screen():
    st.title("BreatheSafe")
    st.caption("Smog & Asthma Risk Predictor")
    
    # 3 Tabs: Log in, Sign up, Forgot Password
    tab_login, tab_signup, tab_forgot = st.tabs(["Log in", "Sign up", "Forgot Password"])
    
    # ---------------- 1. LOGIN TAB ----------------
    with tab_login:
        with st.form("login"):
            u, p = st.text_input("Username"), st.text_input("Password", type="password")
            if st.form_submit_button("Log in", type="primary"):
                user = check_login(u, p)
                if user:
                    st.session_state["user"] = user
                    st.rerun()
                else:
                    st.error("Wrong username or password.")
                    
    # ---------------- 2. SIGN UP TAB (REAL EMAIL OTP) ----------------
    with tab_signup:
        st.subheader("Create a new account")
        
        if "signup_step" not in st.session_state:
            st.session_state["signup_step"] = 1
            
        if st.session_state["signup_step"] == 1:
            with st.form("signup_step1"):
                n = st.text_input("Full name")
                em = st.text_input("Email (e.g., user@example.com)")
                u = st.text_input("Choose a username (3+ chars)")
                p = st.text_input("Password (8+ characters)", type="password")
                p2 = st.text_input("Confirm password", type="password")
                
                submit_signup = st.form_submit_button("Verify Email & Send OTP", type="primary")
                
                if submit_signup:
                    email_regex = r"^[\w\.-]+@[\w\.-]+\.\w+$"
                    
                    if not n.strip():
                        st.error("Full name cannot be empty.")
                    elif not re.match(email_regex, em.strip()):
                        st.error("Please enter a valid email address (must contain '@' and domain).")
                    elif len(u.strip()) < 3:
                        st.error("Username needs at least 3 characters.")
                    elif len(p) < 8:
                        st.error("Password must be at least 8 characters.")
                    elif p != p2:
                        st.error("Passwords do not match.")
                    else:
                        with db() as con:
                            exists = con.execute("SELECT id FROM users WHERE username=?", (u.strip().lower(),)).fetchone()
                        if exists:
                            st.error("That username is already taken. Choose another one.")
                        else:
                            signup_otp = str(random.randint(100000, 999999))
                            with st.spinner("Sending verification OTP to your email..."):
                                sent, msg = send_verification_email(em.strip(), signup_otp, purpose="Account Verification")
                            
                            if sent:
                                st.session_state["temp_signup"] = {
                                    "name": n.strip(),
                                    "email": em.strip(),
                                    "username": u.strip().lower(),
                                    "password": p
                                }
                                st.session_state["signup_otp"] = signup_otp
                                st.session_state["signup_step"] = 2
                                st.rerun()
                            else:
                                st.error(msg)
                            
        elif st.session_state["signup_step"] == 2:
            # LAN/Offline Lab Demo ke liye neeche wali line uncomment kar sakte ho:
            # st.warning(f"🔑 Demo OTP: {st.session_state.get('signup_otp')}")
            
            with st.form("signup_step2"):
                st.info(f"We have sent a 6-digit verification code to **{st.session_state['temp_signup']['email']}**. Please check your inbox (and Spam folder).")
                entered_signup_otp = st.text_input("Enter 6-digit Verification Code")
                
                c1, c2 = st.columns(2)
                verify_signup_btn = c1.form_submit_button("Complete Registration", type="primary")
                back_signup_btn = c2.form_submit_button("Back / Edit Info")
                
                if back_signup_btn:
                    st.session_state["signup_step"] = 1
                    st.rerun()
                    
                if verify_signup_btn:
                    if entered_signup_otp.strip() != st.session_state.get("signup_otp"):
                        st.error("Invalid verification code. Please check your email and try again.")
                    else:
                        data = st.session_state["temp_signup"]
                        ok, msg = register_user(data["username"], data["name"], data["email"], data["password"])
                        if ok:
                            st.success("Email verified & Account created! You can now log in.")
                            st.session_state["signup_step"] = 1
                            st.session_state.pop("temp_signup", None)
                            st.session_state.pop("signup_otp", None)
                        else:
                            st.error(msg)

    # ---------------- 3. FORGOT PASSWORD TAB (REAL EMAIL OTP) ----------------
    with tab_forgot:
        st.subheader("Reset Password via Email OTP")
        
        if "otp_step" not in st.session_state:
            st.session_state["otp_step"] = 1
        
        if st.session_state["otp_step"] == 1:
            with st.form("forgot_step1"):
                f_user = st.text_input("Enter your Username or Email", key="f_user_input")
                send_otp_btn = st.form_submit_button("Send OTP to Email", type="primary")
                
                if send_otp_btn:
                    with db() as con:
                        row = con.execute("SELECT id, username, email FROM users WHERE username=? OR email=?", 
                                          (f_user.strip().lower(), f_user.strip())).fetchone()
                    if row:
                        user_id, uname, user_email = row
                        if not user_email or "@" not in user_email:
                            st.error("No valid email is linked to this account.")
                        else:
                            generated_otp = str(random.randint(100000, 999999))
                            with st.spinner("Sending OTP to your registered email..."):
                                sent, msg = send_verification_email(user_email, generated_otp, purpose="Password Reset")
                            
                            if sent:
                                st.session_state["reset_user_id"] = user_id
                                st.session_state["reset_user_email"] = user_email
                                st.session_state["saved_otp"] = generated_otp
                                st.session_state["otp_step"] = 2
                                st.rerun()
                            else:
                                st.error(msg)
                    else:
                        st.error("No account found with this username or email.")
                        
        elif st.session_state["otp_step"] == 2:
            # LAN/Offline Lab Demo ke liye neeche wali line uncomment kar sakte ho:
            # st.warning(f"🔑 Demo OTP: {st.session_state.get('saved_otp')}")
            
            with st.form("forgot_step2"):
                st.info(f"An OTP has been sent to **{st.session_state.get('reset_user_email')}**. Check your inbox and enter it below.")
                entered_otp = st.text_input("Enter 6-digit OTP")
                new_pass = st.text_input("New Password (8+ characters)", type="password")
                confirm_pass = st.text_input("Confirm New Password", type="password")
                
                c1, c2 = st.columns(2)
                verify_btn = c1.form_submit_button("Reset Password", type="primary")
                back_btn = c2.form_submit_button("Cancel / Back")
                
                if back_btn:
                    st.session_state["otp_step"] = 1
                    st.rerun()
                    
                if verify_btn:
                    if entered_otp.strip() != st.session_state.get("saved_otp"):
                        st.error("Invalid OTP code. Please check your email and try again.")
                    elif len(new_pass) < 8:
                        st.error("Password must be at least 8 characters.")
                    elif new_pass != confirm_pass:
                        st.error("Passwords do not match.")
                    else:
                        salt, pw_hash = hash_password(new_pass)
                        with db() as con:
                            con.execute("UPDATE users SET salt=?, pw_hash=? WHERE id=?", 
                                        (salt, pw_hash, st.session_state["reset_user_id"]))
                        
                        st.success("Password successfully reset! You can now log in using your new password.")
                        st.session_state["otp_step"] = 1
                        st.session_state.pop("saved_otp", None)
                        st.session_state.pop("reset_user_id", None)
                        st.session_state.pop("reset_user_email", None)



def main():
    st.set_page_config(page_title="BreatheSafe", page_icon="🌫️", layout="wide")
    init_db()
    if "user" not in st.session_state:
        auth_screen()
        return
    user = st.session_state["user"]
    st.sidebar.title("BreatheSafe")
    st.sidebar.write(f"Signed in as **{user['name']}**")
    page = st.sidebar.radio("Go to", ["Risk Predictor", "Dashboard", "My History","Edit Profile", "About" ])
    if st.sidebar.button("Log out"):
        st.session_state.clear()
        st.rerun()
    {"Risk Predictor": lambda: page_predictor(user), "Dashboard": page_dashboard,
     "My History": lambda: page_history(user), "About": page_about ,  "Edit Profile":lambda: page_edit_profile(user)  }[page]()


if __name__ == "__main__":
    main()
