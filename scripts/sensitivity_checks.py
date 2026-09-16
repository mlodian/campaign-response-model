"""Two preparation choices, tested against the alternative a reviewer might have used.

1. The 999 "never contacted" code: recency band (as built) vs the raw day count with 999 left in, and vs
   the day count for contacted customers only (0 for the rest) plus a previously-contacted flag.
2. Class balance: no resampling (as built) vs random oversampling of takers to 50/50.

Writes reports/sensitivity_checks.csv. Run after `make pipeline`:
    PYTHONPATH=src python scripts/sensitivity_checks.py
"""

import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from campaign_response.config import load_config
from campaign_response.evaluation import value_per_sale
from campaign_response.modeling import model_ladder
from campaign_response.pipeline import CANDIDATE_NUMERIC, TARGET, split

cfg = load_config()
df = pd.read_parquet(cfg.clean_file)
contacted = df["previously_contacted"] == 1
df["days_if_contacted"] = df["days_since_prev_campaign"].where(contacted, 0)
df["previously_contacted"] = df["previously_contacted"].astype(str)
train, test = split(df, cfg)
y, yt = train[TARGET].to_numpy(), test[TARGET].to_numpy()
selected = json.loads((cfg.reports / "selected_features.json").read_text())["features"]
params = {k: v for k, v in json.loads((cfg.reports / "best_params.json").read_text()).items() if k != "model"}
threshold = json.loads((cfg.reports / "thresholds.json").read_text())["profit_optimal_oof"]
fs, cap_q = cfg["feature_selection"], cfg["outliers"]["cap_quantile"]
seed, folds = cfg["split"]["random_state"], cfg["cv"]["folds"]
campaign = cfg["campaign"]
v, c = value_per_sale(campaign), -campaign["cost_not_taken_up"]
MODELS = ["Logistic scorecard (WoE)", "Logistic regression (one-hot)", "Random forest", "LightGBM (tuned)"]


def ladder(feats, numeric):
    lad = model_ladder(feats, numeric, fs, cap_q)
    lad["LightGBM (tuned)"] = clone(lad["LightGBM"]).set_params(**params)
    return lad


def profit(p):
    call = p >= threshold
    return ((call & (yt == 1)).sum() * v - (call & (yt == 0)).sum() * c) / len(yt) * campaign["list_size"]


def scores(pipe, feats, idx=None):
    idx = np.arange(len(y)) if idx is None else idx
    p = clone(pipe).fit(train[feats].iloc[idx], y[idx]).predict_proba(test[feats])[:, 1]
    return {"test_pr_auc": average_precision_score(yt, p), "test_roc_auc": roc_auc_score(yt, p),
            "test_brier": brier_score_loss(yt, p), "mean_score": p.mean(), "call_rate": (p >= threshold).mean(),
            "profit_per_list": profit(p)}


rows = []
raw = ["days_since_prev_campaign" if f == "prev_contact_recency" else f for f in selected]
flagged = [f for f in selected if f != "prev_contact_recency"] + ["days_if_contacted", "previously_contacted"]
cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
for variant, feats, numeric in [("Recency band (as built)", selected, CANDIDATE_NUMERIC),
                                ("999 kept as a number of days", raw, CANDIDATE_NUMERIC + ["days_since_prev_campaign"]),
                                ("Days if contacted + contacted flag", flagged, CANDIDATE_NUMERIC + ["days_if_contacted"])]:
    lad = ladder(feats, numeric)
    for name in MODELS:
        cv_pr = [average_precision_score(y[va], clone(lad[name]).fit(train[feats].iloc[tr], y[tr])
                                         .predict_proba(train[feats].iloc[va])[:, 1])
                 for tr, va in cv.split(train[feats], y)]
        rows.append({"check": "999 sentinel", "variant": variant, "model": name, "cv_pr_auc": np.mean(cv_pr),
                     **scores(lad[name], feats)})

rng = np.random.default_rng(seed)
pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
oversampled = np.concatenate([neg, rng.choice(pos, len(neg), replace=True)])
lgbm = ladder(selected, CANDIDATE_NUMERIC)["LightGBM (tuned)"]
for variant, idx in [("No resampling (as built)", None), ("Takers oversampled to 50/50", oversampled)]:
    rows.append({"check": "class balance", "variant": variant, "model": "LightGBM (tuned)",
                 **scores(lgbm, selected, idx)})

out = pd.DataFrame(rows)
out.to_csv(cfg.reports / "sensitivity_checks.csv", index=False)
print(out.round(4).to_string(index=False))
