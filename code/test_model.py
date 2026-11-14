import joblib, json, pandas as pd
meta = json.load(open("../models/model_meta.json"))
model = joblib.load("../models/best_model.joblib")
X = pd.read_csv("../data/processed/engineered_features.csv")[meta["features"]].iloc[[0]]
X["pm2_5"] = 220      # try 30, 100, 220
p = model.predict_proba(X)[0]
print(dict(zip(meta["class_names"], p.round(3))))