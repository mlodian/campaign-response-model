"""Model ladder, cross-validation, tuning and overfitting diagnostics."""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_validate, learning_curve,
)
from sklearn.pipeline import Pipeline

from .evaluation import ks_statistic, lift_at
from .features import onehot_preprocessor, split_types, woe_preprocessor

log = logging.getLogger(__name__)


def _lift10(y, p):
    return lift_at(np.asarray(y), np.asarray(p), 0.10)


def _ks(y, p):
    return ks_statistic(np.asarray(y), np.asarray(p))


def scorers():
    from sklearn.metrics import make_scorer

    return {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "brier": "neg_brier_score",
        "ks": make_scorer(_ks, response_method="predict_proba"),
        "lift_top10": make_scorer(_lift10, response_method="predict_proba"),
    }


class PreviousOutcomeRule(ClassifierMixin, BaseEstimator):
    """Rule a call centre could use without a model: previous success first,
    then previously contacted, then mobile contacts."""

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        prev = X["previous_outcome"].astype(str)
        score = (0.6 * prev.str.endswith("_s***s") + 0.2 * prev.str.endswith("_f***e")
                 + 0.2 * X["contact_medium"].astype(str).str.endswith("_c***r")).to_numpy(float)
        return np.column_stack([1 - score, score])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def lgbm(**params) -> LGBMClassifier:
    base = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=42, n_jobs=-1, verbose=-1)
    base.update(params)
    return LGBMClassifier(**base)


def model_ladder(features, numeric_all, fs_cfg, cap_q) -> dict[str, Pipeline]:
    num, cat = split_types(features, numeric_all)
    rare = fs_cfg["rare_category_share"]
    return {
        "Baseline (response rate)": Pipeline([("model", DummyClassifier(strategy="prior"))]),
        "Business rule (previous outcome)": Pipeline([("model", PreviousOutcomeRule())]),
        "Logistic scorecard (WoE)": Pipeline([
            ("prep", woe_preprocessor(num, cat, rare, cap_q)),
            ("model", LogisticRegression(max_iter=2000, C=1.0)),
        ]),
        "Logistic regression (one-hot)": Pipeline([
            ("prep", onehot_preprocessor(num, cat, rare, cap_q, scale=True)),
            ("model", LogisticRegression(max_iter=3000, C=0.5)),
        ]),
        "Random forest": Pipeline([
            ("prep", onehot_preprocessor(num, cat, rare, cap_q, scale=False)),
            ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=20, max_features="sqrt",
                                             n_jobs=-1, random_state=42)),
        ]),
        "LightGBM": Pipeline([
            ("prep", onehot_preprocessor(num, cat, rare, cap_q, scale=False)),
            ("model", lgbm()),
        ]),
    }


def cross_validate_ladder(ladder, X, y, folds, seed) -> pd.DataFrame:
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    rows = []
    for name, pipe in ladder.items():
        t = time.time()
        res = cross_validate(pipe, X, y, cv=cv, scoring=scorers(), return_train_score=True,
                             n_jobs=1, error_score="raise")
        for i in range(folds):
            rows.append({"model": name, "fold": i,
                         **{m: res[f"test_{m}"][i] for m in scorers()},
                         **{f"train_{m}": res[f"train_{m}"][i] for m in ("pr_auc", "roc_auc")}})
        log.info("CV %-34s PR-AUC %.3f (%.0fs)", name, res["test_pr_auc"].mean(), time.time() - t)
    out = pd.DataFrame(rows)
    out["brier"] = -out["brier"]
    return out


def summarise_cv(folds: pd.DataFrame) -> pd.DataFrame:
    metrics = ["pr_auc", "roc_auc", "ks", "lift_top10", "brier", "train_pr_auc", "train_roc_auc"]
    agg = folds.groupby("model", sort=False)[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg["gini_mean"] = 2 * agg["roc_auc_mean"] - 1
    agg["overfit_gap_pr_auc"] = agg["train_pr_auc_mean"] - agg["pr_auc_mean"]
    return agg


def paired_differences(folds: pd.DataFrame, reference: str, metric: str = "pr_auc") -> pd.DataFrame:
    """Per-fold difference between the reference model and each other model (same folds)."""
    wide = folds.pivot(index="fold", columns="model", values=metric)
    diff = wide.drop(columns=reference).rsub(wide[reference], axis=0)
    return pd.DataFrame({"model": diff.columns, f"{metric}_diff_mean": diff.mean().to_numpy(),
                         f"{metric}_diff_std": diff.std().to_numpy(),
                         "reference_wins": (diff > 0).sum().to_numpy(), "folds": len(diff)}) \
        .sort_values(f"{metric}_diff_mean").reset_index(drop=True)


def tune(pipe: Pipeline, name: str, X, y, n_iter, folds, seed) -> RandomizedSearchCV:
    """Randomised search on PR-AUC. Tree depth, leaf size and regularisation are
    searched together because they jointly control over- and underfitting."""
    if "LightGBM" in name:
        space = {
            "model__n_estimators": randint(150, 900),
            "model__learning_rate": loguniform(0.01, 0.1),
            "model__num_leaves": randint(8, 64),
            "model__min_child_samples": randint(20, 200),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.5, 0.5),
            "model__reg_lambda": loguniform(0.1, 30),
        }
    elif "forest" in name:
        space = {
            "model__n_estimators": randint(200, 800),
            "model__min_samples_leaf": randint(5, 80),
            "model__max_features": uniform(0.2, 0.6),
            "model__max_depth": [None, 8, 12, 16],
        }
    else:
        space = {"model__C": loguniform(0.005, 10)}
    search = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring="average_precision", refit=True,
                                cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
                                random_state=seed, n_jobs=1, error_score="raise")
    t = time.time()
    search.fit(X, y)
    log.info("Tuned %s in %.0fs: CV PR-AUC %.4f", name, time.time() - t, search.best_score_)
    return search


def out_of_fold(pipe, X, y, folds, seed) -> np.ndarray:
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]


def learning_curve_table(pipe, X, y, folds, seed) -> pd.DataFrame:
    sizes, train, valid = learning_curve(
        pipe, X, y, train_sizes=np.linspace(0.1, 1.0, 6), scoring="average_precision",
        cv=StratifiedKFold(folds, shuffle=True, random_state=seed), n_jobs=1, shuffle=True, random_state=seed,
    )
    return pd.DataFrame({"train_size": sizes, "train_mean": train.mean(1), "train_std": train.std(1),
                         "valid_mean": valid.mean(1), "valid_std": valid.std(1)})


def boosting_curve(pipe: Pipeline, X, y, seed) -> pd.DataFrame:
    """Train/validation log-loss by boosting round on an inner 80/20 split."""
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split

    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    p = clone(pipe)
    prep = p.named_steps["prep"].fit(Xa, ya)
    model = p.named_steps["model"]
    model.fit(prep.transform(Xa), ya, eval_set=[(prep.transform(Xa), ya), (prep.transform(Xb), yb)],
              eval_names=["train", "validation"], eval_metric="binary_logloss")
    ev = model.evals_result_
    return pd.DataFrame({"round": np.arange(1, len(ev["train"]["binary_logloss"]) + 1),
                         "train_logloss": ev["train"]["binary_logloss"],
                         "valid_logloss": ev["validation"]["binary_logloss"]})


def leakage_benchmark(features, numeric_all, fs_cfg, cap_q, X, y, folds, seed) -> pd.DataFrame:
    """CV LightGBM with and without the post-call duration field."""
    rows = []
    for name, feats in [("Without call duration (deployable)", features),
                        ("With call duration (leaky)", features + ["leak_call_duration_s"])]:
        pipe = model_ladder(feats, numeric_all + ["leak_call_duration_s"], fs_cfg, cap_q)["LightGBM"]
        res = cross_validate(pipe, X[feats], y, cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
                             scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"}, error_score="raise")
        rows.append({"variant": name, "pr_auc": res["test_pr_auc"].mean(), "roc_auc": res["test_roc_auc"].mean()})
        log.info("Leakage benchmark %-36s PR-AUC %.3f", name, rows[-1]["pr_auc"])
    return pd.DataFrame(rows)
