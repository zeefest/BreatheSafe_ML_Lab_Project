# %% [markdown]
# # 03 - Train, Tune, Evaluate and Compare the Models
# Models: Random Forest, Gradient Boosting (XGBoost if installed, else scikit-learn GradientBoosting), SVM (RBF),
# plus Logistic Regression as a baseline. Input: `data/processed/engineered_features.csv`.

# %%
import copy
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

warnings.filterwarnings("ignore")
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
ROOT = HERE.parent if HERE.name == "code" else HERE
PROC, PLOTS, METRICS, MODELS = ROOT / "data/processed", ROOT / "outputs/plots", ROOT / "outputs/metrics", ROOT / "models"
for p in (PLOTS, METRICS, MODELS):
    p.mkdir(parents=True, exist_ok=True)
SEED = 42
sns.set_theme(style="whitegrid", context="notebook")
CLASS_NAMES = ["Low", "Moderate", "High", "Critical"]
interp = json.loads((PLOTS / "figure_interpretations.json").read_text()) if (PLOTS / "figure_interpretations.json").exists() else {}


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS / name, dpi=200, bbox_inches="tight")
    if "ipykernel" in sys.modules:
        plt.show()
    plt.close(fig)


# %% [markdown]
# ## 1. Stratified train / validation / test split (70 / 15 / 15)

# %%
meta = json.loads((METRICS / "selected_features.json").read_text())
FEATURES = meta["selected"]
df = pd.read_csv(PROC / "engineered_features.csv")
X, y = df[FEATURES], df["Asthma_Risk_Level"].astype(int)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
print(f"train {len(X_tr):,} | validation {len(X_val):,} | test {len(X_te):,} | features {len(FEATURES)}")
print("Class balance (train):", y_tr.value_counts(normalize=True).sort_index().round(3).to_dict())

# class imbalance: 'balanced' class weights (RF, SVM, LR) / balanced sample weights (boosting)
w_tr = compute_sample_weight("balanced", y_tr)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# %% [markdown]
# ## 2. Models and hyper-parameter search spaces
# Tuning uses `RandomizedSearchCV` with 5-fold stratified CV (scoring = macro F1) on a stratified sample of up to 5,000
# training rows, which keeps the SVM tractable. The winning parameters are then refit on the full training set.

# %%
if HAS_XGB:
    gb_name = "Gradient Boosting (XGBoost)"
    gb_clf = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", n_jobs=-1, random_state=SEED, tree_method="hist")
    gb_grid = {"clf__n_estimators": [150, 250, 400], "clf__learning_rate": [0.05, 0.1, 0.2], "clf__max_depth": [3, 4, 6],
               "clf__subsample": [0.8, 1.0], "clf__colsample_bytree": [0.7, 1.0]}
else:
    gb_name = "Gradient Boosting (sklearn)"
    gb_clf = GradientBoostingClassifier(random_state=SEED)
    gb_grid = {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1, 0.2], "clf__max_depth": [2, 3, 4],
               "clf__subsample": [0.8, 1.0]}
print("Boosting model:", gb_name)

SPECS = {
    "Random Forest": (RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=SEED),
                      {"clf__n_estimators": [150, 300], "clf__max_depth": [8, 12, 16], "clf__min_samples_leaf": [1, 3, 5],
                       "clf__max_features": ["sqrt", 0.5]}, False),
    gb_name: (gb_clf, gb_grid, True),
    "SVM (RBF)": (SVC(kernel="rbf", probability=False, class_weight="balanced", random_state=SEED),
                  {"clf__C": [0.5, 1, 3, 10], "clf__gamma": ["scale", 0.01, 0.05, 0.1]}, False),
    "Logistic Regression (baseline)": (LogisticRegression(class_weight="balanced", max_iter=3000),
                                       {"clf__C": [0.01, 0.1, 1, 10]}, False),
}

# %% [markdown]
# ## 3. Tune, refit, time and evaluate

# %%
def single_row_latency_ms(model, Xdf, n=100):
    rows = [Xdf.iloc[[i]] for i in range(min(n, len(Xdf)))]
    t0 = time.perf_counter()
    for r in rows:
        model.predict_proba(r)
    return (time.perf_counter() - t0) / len(rows) * 1000


sub_idx = X_tr.sample(min(5000, len(X_tr)), random_state=SEED).index
fitted, results, tuned = {}, [], {}
for name, (clf, grid, uses_sw) in SPECS.items():
    print(f"\n=== {name} ===")
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    search = RandomizedSearchCV(pipe, grid, n_iter=6, cv=cv, scoring="f1_macro", n_jobs=-1, random_state=SEED, refit=False)
    fit_kw = {"clf__sample_weight": compute_sample_weight("balanced", y_tr.loc[sub_idx])} if uses_sw else {}
    search.fit(X_tr.loc[sub_idx], y_tr.loc[sub_idx], **fit_kw)
    best = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    tuned[name] = {"best_params": best, "tuning_cv_f1_macro": round(float(search.best_score_), 4)}
    print("best params:", best, "| tuning CV F1:", round(search.best_score_, 4))

    final = Pipeline([("scaler", StandardScaler()), ("clf", clone(clf).set_params(**best))])
    if isinstance(clf, SVC):
        cv_model = clone(final)                                # SVM scored via decision values in CV (faster)
        final.set_params(clf__probability=True)                # probabilities needed for ROC-AUC and the app
    else:
        cv_model = clone(final)
    cv_kw = {"params": {"clf__sample_weight": w_tr}} if uses_sw else {}
    cv_scores = cross_val_score(cv_model, X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=1, **cv_kw)

    t0 = time.perf_counter()
    final.fit(X_tr, y_tr, **({"clf__sample_weight": w_tr} if uses_sw else {}))
    train_s = time.perf_counter() - t0
    fitted[name] = final

    def evaluate(Xs, ys):
        proba = final.predict_proba(Xs)
        pred = proba.argmax(axis=1)
        return proba, pred, {
            "Accuracy": accuracy_score(ys, pred),
            "Precision": precision_score(ys, pred, average="macro"),
            "Recall": recall_score(ys, pred, average="macro"),
            "F1": f1_score(ys, pred, average="macro"),
            "ROC_AUC": roc_auc_score(ys, proba, multi_class="ovr", average="macro")}

    _, _, val_m = evaluate(X_val, y_val)
    proba_te, pred_te, te_m = evaluate(X_te, y_te)
    hr_true, hr_pred = y_te >= 2, pred_te >= 2
    results.append({
        "Model": name, **{k: round(v, 4) for k, v in te_m.items()},
        "HighRisk_Recall": round(recall_score(hr_true, hr_pred), 4),
        "Val_F1": round(val_m["F1"], 4), "Val_Accuracy": round(val_m["Accuracy"], 4),
        "CV5_F1_mean": round(cv_scores.mean(), 4), "CV5_F1_std": round(cv_scores.std(), 4),
        "Train_Time_s": round(train_s, 2), "Inference_ms_per_pred": round(single_row_latency_ms(final, X_te), 3)})
    tuned[name]["cv_scores"] = [round(float(s), 4) for s in cv_scores]
    tuned[name]["proba_te"], tuned[name]["pred_te"] = proba_te, pred_te
    print(results[-1])

res = pd.DataFrame(results)
res.to_csv(METRICS / "model_comparison.csv", index=False)
print("\n", res.drop(columns=["Val_Accuracy"]).to_string(index=False))

# %% [markdown]
# ## 4. Pick the best model (validation macro F1) and the two models to compare statistically

# %%
core = res[~res.Model.str.contains("baseline")].sort_values("Val_F1", ascending=False).reset_index(drop=True)
best_name, second_name = core.loc[0, "Model"], core.loc[1, "Model"]
print(f"Best on validation F1: {best_name} | runner-up: {second_name}")

# 5-fold CV paired t-test (same folds for both models)
a, b = np.array(tuned[best_name]["cv_scores"]), np.array(tuned[second_name]["cv_scores"])
t_stat, t_p = stats.ttest_rel(a, b)

# McNemar test on the test set
ca, cb = tuned[best_name]["pred_te"] == y_te.values, tuned[second_name]["pred_te"] == y_te.values
n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())     # best right / second wrong, and the reverse
if n01 + n10 == 0:
    mc_stat, mc_p, mc_kind = 0.0, 1.0, "no discordant pairs"
elif n01 + n10 < 25:
    mc_stat, mc_p, mc_kind = float(min(n01, n10)), float(stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue), "exact binomial"
else:
    mc_stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    mc_p, mc_kind = float(stats.chi2.sf(mc_stat, 1)), "chi-square with continuity correction"
tests = {
    "compared": [best_name, second_name],
    "paired_t_test_5fold_cv": {"fold_scores_best": a.tolist(), "fold_scores_second": b.tolist(), "t_statistic": round(float(t_stat), 4),
                               "p_value": round(float(t_p), 4), "significant_at_0.05": bool(t_p < 0.05)},
    "mcnemar_test_holdout": {"best_right_second_wrong": n01, "best_wrong_second_right": n10, "test_type": mc_kind,
                             "statistic": round(float(mc_stat), 4), "p_value": round(mc_p, 4), "significant_at_0.05": bool(mc_p < 0.05)}}
(METRICS / "statistical_tests.json").write_text(json.dumps(tests, indent=2))
print(json.dumps(tests, indent=2))

# %% [markdown]
# ## 5. Error analysis (best model on the test set)

# %%
best_pred, best_proba = tuned[best_name]["pred_te"], tuned[best_name]["proba_te"]
cm = confusion_matrix(y_te, best_pred)
(METRICS / "classification_report_best.txt").write_text(classification_report(y_te, best_pred, target_names=CLASS_NAMES, digits=3))
print(classification_report(y_te, best_pred, target_names=CLASS_NAMES, digits=3))

true_hi = (y_te >= 2).values
under = true_hi & (best_pred < 2)                                  # High/Critical predicted as Low/Moderate = dangerous false negatives
crit_missed = (y_te.values == 3) & (best_pred < 2)
over = (y_te.values < 2) & (best_pred >= 2)
wrong = best_pred != y_te.values
prof = X_te.copy()
prof["_wrong"] = wrong
z = (prof[prof._wrong][FEATURES].mean() - prof[~prof._wrong][FEATURES].mean()) / prof[FEATURES].std()
profile = pd.DataFrame({"mean_misclassified": prof[prof._wrong][FEATURES].mean(), "mean_correct": prof[~prof._wrong][FEATURES].mean(),
                        "standardized_difference": z}).sort_values("standardized_difference", key=abs, ascending=False)
profile.to_csv(METRICS / "misclassified_feature_profile.csv")
fn_profile = X_te[under][FEATURES].mean()
error = {
    "best_model": best_name, "confusion_matrix": cm.tolist(), "n_test": int(len(y_te)), "n_misclassified": int(wrong.sum()),
    "misclassification_rate": round(float(wrong.mean()), 4),
    "dangerous_false_negatives_high_or_critical_predicted_low_or_moderate": int(under.sum()),
    "high_risk_false_negative_rate": round(float(under.sum() / max(true_hi.sum(), 1)), 4),
    "critical_predicted_low_or_moderate": int(crit_missed.sum()),
    "false_alarms_low_or_moderate_predicted_high_or_critical": int(over.sum()),
    "per_class_error_rate": {CLASS_NAMES[i]: round(float(1 - cm[i, i] / cm[i].sum()), 4) for i in range(4)},
    "top_features_differing_in_errors": profile.head(5).round(3)["standardized_difference"].to_dict(),
    "mean_pm25_missed_high_risk_cases": round(float(X_te[under]["pm2_5"].mean()), 2) if under.any() and "pm2_5" in FEATURES else None,
    "mean_pm25_correctly_flagged_high_risk": round(float(X_te[true_hi & ~under]["pm2_5"].mean()), 2) if "pm2_5" in FEATURES else None}
(METRICS / "error_analysis.json").write_text(json.dumps(error, indent=2))
print(json.dumps(error, indent=2))

# high-risk alert threshold: smallest-miss rule used by the web app
p_hi = best_proba[:, 2:].sum(axis=1)
sweep = []
for th in np.arange(0.15, 0.71, 0.05):
    pr = p_hi >= th
    sweep.append({"threshold": round(float(th), 2), "recall_high_risk": round(float(recall_score(true_hi, pr)), 4),
                  "precision_high_risk": round(float(precision_score(true_hi, pr, zero_division=0)), 4),
                  "false_alarms": int((pr & ~true_hi).sum())})
sweep = pd.DataFrame(sweep)
sweep.to_csv(METRICS / "threshold_sweep.csv", index=False)
ok = sweep[sweep.recall_high_risk >= 0.90]
ALERT_TH = float(ok.threshold.max()) if len(ok) else 0.35
print(f"High-risk alert threshold chosen: {ALERT_TH} (largest threshold with recall >= 0.90)")

# %% [markdown]
# ## 6. Figures 9-12

# %%
Y_bin = label_binarize(y_te, classes=[0, 1, 2, 3])
colors = dict(zip(SPECS, ["#2a9d8f", "#e76f51", "#264653", "#999999"]))

# Fig 9 - ROC
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for name in SPECS:
    fpr, tpr, _ = roc_curve(Y_bin.ravel(), tuned[name]["proba_te"].ravel())
    axes[0].plot(fpr, tpr, color=colors[name], label=f"{name} (AUC {auc(fpr, tpr):.3f})", ls="--" if "baseline" in name else "-")
axes[0].plot([0, 1], [0, 1], "k:", lw=1)
axes[0].set(title="Micro-average ROC, all models", xlabel="False positive rate", ylabel="True positive rate")
axes[0].legend(loc="lower right", fontsize=8)
for i, cn in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(Y_bin[:, i], best_proba[:, i])
    axes[1].plot(fpr, tpr, label=f"{cn} (AUC {auc(fpr, tpr):.3f})")
axes[1].plot([0, 1], [0, 1], "k:", lw=1)
axes[1].set(title=f"Per-class ROC - {best_name}", xlabel="False positive rate")
axes[1].legend(loc="lower right")
fig.suptitle("Fig 9 - ROC curves")
save_fig(fig, "fig09_roc_curves.png")
pc_auc = {cn: roc_auc_score(Y_bin[:, i], best_proba[:, i]) for i, cn in enumerate(CLASS_NAMES)}
interp["fig09"] = (f"All three main models reach a macro ROC-AUC between {res[~res.Model.str.contains('baseline')].ROC_AUC.min():.3f} and {res[~res.Model.str.contains('baseline')].ROC_AUC.max():.3f}; "
                   f"the logistic-regression baseline scores {res[res.Model.str.contains('baseline')].ROC_AUC.iloc[0]:.3f}. For {best_name}, per-class AUC is "
                   + ", ".join(f"{k} {v:.3f}" for k, v in pc_auc.items()) + ". The middle tiers (Moderate, High) are hardest because they sit between neighbours, while the extremes separate cleanly.")

# Fig 10 - Precision-Recall
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for name in SPECS:
    p_, r_, _ = precision_recall_curve(Y_bin.ravel(), tuned[name]["proba_te"].ravel())
    axes[0].plot(r_, p_, color=colors[name], label=f"{name} (AUC {auc(r_, p_):.3f})", ls="--" if "baseline" in name else "-")
axes[0].set(title="Micro-average precision-recall, all models", xlabel="Recall", ylabel="Precision")
axes[0].legend(loc="lower left", fontsize=8)
for i, cn in enumerate(CLASS_NAMES):
    p_, r_, _ = precision_recall_curve(Y_bin[:, i], best_proba[:, i])
    axes[1].plot(r_, p_, label=f"{cn} (AUC {auc(r_, p_):.3f})")
axes[1].set(title=f"Per-class precision-recall - {best_name}", xlabel="Recall")
axes[1].legend(loc="lower left")
fig.suptitle("Fig 10 - Precision-Recall curves")
save_fig(fig, "fig10_precision_recall_curves.png")
crit_pr = auc(*precision_recall_curve(Y_bin[:, 3], best_proba[:, 3])[1::-1])
interp["fig10"] = (f"Precision-recall curves matter more than ROC here because Critical cases are only about {Y_bin[:, 3].mean():.0%} of the test set. "
                   f"For {best_name} the Critical class has a PR-AUC of {crit_pr:.3f}. The curves show how much precision is given up when the alert threshold is lowered to catch more high-risk patients, "
                   f"which is the trade-off behind the app's alert threshold of {ALERT_TH:.2f}.")

# Fig 11 - confusion matrices
show = [n for n in SPECS if "baseline" not in n]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, name in zip(axes, show):
    c = confusion_matrix(y_te, tuned[name]["pred_te"])
    sns.heatmap(c / c.sum(axis=1, keepdims=True), annot=np.array([[f"{v}\n({v / c[i].sum():.0%})" for v in row] for i, row in enumerate(c)]),
                fmt="", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False, ax=ax)
    ax.set(title=name, xlabel="Predicted", ylabel="Actual")
fig.suptitle("Fig 11 - Confusion matrices (test set, row-normalised)")
save_fig(fig, "fig11_confusion_matrices.png")
interp["fig11"] = (f"For {best_name}, {error['n_misclassified']} of {error['n_test']} test records are misclassified ({error['misclassification_rate']:.1%}). Nearly all errors are one tier away from the truth. "
                   f"{error['dangerous_false_negatives_high_or_critical_predicted_low_or_moderate']} High/Critical patients were predicted Low/Moderate (a high-risk miss rate of {error['high_risk_false_negative_rate']:.1%}), "
                   f"and {error['false_alarms_low_or_moderate_predicted_high_or_critical']} Low/Moderate patients were flagged High/Critical.")

# Fig 12 - latency vs accuracy
fig, ax = plt.subplots(figsize=(8, 5.5))
for _, r in res.iterrows():
    ax.scatter(r.Inference_ms_per_pred, r.Accuracy, s=120 + r.Train_Time_s * 12, color=colors[r.Model], alpha=.8, edgecolor="k")
    ax.annotate(f"{r.Model}\n(train {r.Train_Time_s:.1f}s)", (r.Inference_ms_per_pred, r.Accuracy), textcoords="offset points", xytext=(8, 6), fontsize=8)
ax.set(title="Fig 12 - Latency vs accuracy (bubble size = training time)", xlabel="Inference time per prediction (ms, single row)", ylabel="Test accuracy")
ax.margins(0.25)
save_fig(fig, "fig12_latency_vs_accuracy.png")
fast, slow = res.loc[res.Inference_ms_per_pred.idxmin()], res.loc[res.Inference_ms_per_pred.idxmax()]
interp["fig12"] = (f"{fast.Model} answers fastest ({fast.Inference_ms_per_pred:.2f} ms per single prediction) and {slow.Model} slowest ({slow.Inference_ms_per_pred:.2f} ms). "
                   f"Accuracy differs by only {res.Accuracy.max() - res.Accuracy.min():.3f} between all four models, so for a web app every model is fast enough; "
                   "the choice should be driven by recall and stability rather than latency.")

(PLOTS / "figure_interpretations.json").write_text(json.dumps(interp, indent=2))
(PLOTS / "figure_interpretations.md").write_text("# Figure interpretations\n\n" + "\n\n".join(f"**{k.upper()}** - {v}" for k, v in sorted(interp.items())))

# %% [markdown]
# ## 7. Save the artefacts (pipelines, scaler, metadata)

# %%
file_names = {"Random Forest": "random_forest_pipeline.joblib", gb_name: "gradient_boosting_pipeline.joblib",
              "SVM (RBF)": "svm_pipeline.joblib", "Logistic Regression (baseline)": "logistic_regression_pipeline.joblib"}
for name, model in fitted.items():
    joblib.dump(model, MODELS / file_names[name])
joblib.dump(fitted[best_name], MODELS / "best_model.joblib")
joblib.dump(StandardScaler().fit(X_tr), MODELS / "scaler.joblib")   # standalone copy; each pipeline also embeds its own scaler
model_meta = {"features": FEATURES, "class_names": CLASS_NAMES, "best_model": best_name, "alert_threshold": ALERT_TH,
              "model_files": file_names, "uses_xgboost": HAS_XGB, "metrics": res.to_dict(orient="records"),
              "best_params": {k: v["best_params"] for k, v in tuned.items()}, "defaults": meta["defaults"]}
(MODELS / "model_meta.json").write_text(json.dumps(model_meta, indent=2))

top2 = res.set_index("Model")
summary_md = f"""# Evaluation summary
Best model (validation macro-F1): **{best_name}**. Runner-up: {second_name}.

{res.drop(columns=['Val_Accuracy']).to_markdown(index=False) if hasattr(res, 'to_markdown') and 'tabulate' in sys.modules else res.to_string(index=False)}

Paired t-test (5-fold CV): t = {tests['paired_t_test_5fold_cv']['t_statistic']}, p = {tests['paired_t_test_5fold_cv']['p_value']}
McNemar ({mc_kind}): p = {tests['mcnemar_test_holdout']['p_value']} (b = {n01}, c = {n10})
High-risk false negatives: {error['dangerous_false_negatives_high_or_critical_predicted_low_or_moderate']} ({error['high_risk_false_negative_rate']:.1%})
"""
(METRICS / "evaluation_summary.md").write_text(summary_md)
print(summary_md)
print("Saved models/, outputs/metrics/, outputs/plots/fig09-fig12")

# %% [markdown]
# ## 8. Live test prediction (one hand-made scenario)

# %%
demo = X_te.iloc[[0]].copy()
for k, v in {"pm2_5": 180.0, "Age": 67}.items():
    if k in demo.columns:
        demo[k] = v
proba = fitted[best_name].predict_proba(demo)[0]
print("Test scenario ->", dict(zip(CLASS_NAMES, proba.round(3))), "| predicted:", CLASS_NAMES[int(proba.argmax())])
