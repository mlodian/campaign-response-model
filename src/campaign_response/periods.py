"""Campaign-period checks.

The file has no dates, but the four monthly macro indicators take one value
combination per campaign period. That makes them period labels: a random split
puts the same periods in train and test, and much of the pooled ranking power
comes from telling good months from bad ones. A real list is called within one
period, so the model is also judged here on
  - grouped CV that holds out whole periods, and
  - ranking *within* a period (weighted mean ROC-AUC and top-decile lift).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from . import evaluation as ev
from . import plotting as P

PERIOD_KEYS = ["employment_variation_rate", "consumer_price_index", "consumer_confidence_index",
               "number_of_employees"]
TIMING_FEATURES = PERIOD_KEYS + ["interbank_rate_3m", "last_contact_month"]
MIN_CLASS = 10   # periods with fewer takers or non-takers are too small to rank within


def period_ids(df: pd.DataFrame) -> pd.Series:
    return df.groupby(PERIOD_KEYS, sort=True).ngroup().rename("period")


def within_period(y: np.ndarray, p: np.ndarray, period: np.ndarray) -> dict:
    rows = []
    for g in np.unique(period):
        m = period == g
        yy, pp = y[m], p[m]
        if yy.sum() >= MIN_CLASS and (1 - yy).sum() >= MIN_CLASS:
            rows.append({"n": m.sum(), "roc_auc": roc_auc_score(yy, pp), "lift_top10": ev.lift_at(yy, pp, 0.1)})
    r = pd.DataFrame(rows)
    w = r["n"] / r["n"].sum()
    return {"periods_scored": len(r), "within_roc_auc": float((r["roc_auc"] * w).sum()),
            "within_lift_top10": float((r["lift_top10"] * w).sum())}


def grouped_oof(pipe, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, folds: int, seed: int) -> np.ndarray:
    oof = np.zeros(len(y))
    cv = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
    for tr, va in cv.split(X, y, groups):
        oof[va] = clone(pipe).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
    return oof


def validation_table(variants: dict, train: pd.DataFrame, test: pd.DataFrame, target: str,
                     folds: int, seed: int) -> pd.DataFrame:
    """variants: name -> (pipeline, feature list). Returns one row per variant and scheme."""
    y, yt = train[target].to_numpy(), test[target].to_numpy()
    g, gt = period_ids_for(train), period_ids_for(test)
    rows = []
    for name, (pipe, feats) in variants.items():
        oof = grouped_oof(pipe, train[feats], y, g, folds, seed)
        rows.append({"model": name, "scheme": "Grouped CV (unseen periods)",
                     "roc_auc": roc_auc_score(y, oof), "pr_auc": average_precision_score(y, oof),
                     "lift_top10": ev.lift_at(y, oof, 0.1), **within_period(y, oof, g)})
        p = clone(pipe).fit(train[feats], y).predict_proba(test[feats])[:, 1]
        rows.append({"model": name, "scheme": "Random split test (seen periods)",
                     "roc_auc": roc_auc_score(yt, p), "pr_auc": average_precision_score(yt, p),
                     "lift_top10": ev.lift_at(yt, p, 0.1), **within_period(yt, p, gt)})
    return pd.DataFrame(rows)


def period_ids_for(df: pd.DataFrame) -> np.ndarray:
    return df["period"].to_numpy() if "period" in df else period_ids(df).to_numpy()


def period_breakdown(test: pd.DataFrame, y: np.ndarray, p: np.ndarray, threshold: float,
                     campaign: dict) -> pd.DataFrame:
    """Per-period take-up, share called and profit per 10,000 customers on the test set."""
    v, c = ev.value_per_sale(campaign), -campaign["cost_not_taken_up"]
    call = p >= threshold
    t = pd.DataFrame({"period": period_ids_for(test), "month": test["last_contact_month"].astype(str).to_numpy(),
                      "number_of_employees": test["number_of_employees"].to_numpy(),
                      "y": y, "call": call, "tp": call & (y == 1), "fp": call & (y == 0)})
    out = t.groupby("period").agg(month=("month", "first"), number_of_employees=("number_of_employees", "first"),
                                  customers=("y", "size"), take_up_rate=("y", "mean"), call_rate=("call", "mean"),
                                  tp=("tp", "sum"), fp=("fp", "sum")).reset_index()
    size = campaign["list_size"]
    out["profit_per_10k_model"] = (out["tp"] * v - out["fp"] * c) / out["customers"] * size
    out["profit_per_10k_call_all"] = (out["take_up_rate"] * v - (1 - out["take_up_rate"]) * c) * size
    return out.sort_values("number_of_employees").reset_index(drop=True)


def plot_breakdown(b: pd.DataFrame, break_even: float, path: Path) -> Path:
    b = b[b["customers"] >= 100].reset_index(drop=True)
    fig, ax = P.new_figure(10, 4.6)
    x = np.arange(len(b))
    ax.bar(x, b["take_up_rate"], width=0.6, color=[P.BLUE if r >= break_even else P.CONTEXT
                                                  for r in b["take_up_rate"]], linewidth=0)
    ax.plot(x, b["call_rate"], color=P.INK, marker="o", markersize=4, linewidth=1.5)
    ax.axhline(break_even, color=P.INK_2, linewidth=1, linestyle="--")
    ax.text(len(b) - 0.5, break_even + 0.01, f"break-even take-up {break_even:.2f}", ha="right",
            fontsize=9.5, color=P.INK_2)
    ax.text(x[0], b["call_rate"].iloc[0] + 0.03, "share of customers the model calls", fontsize=9.5)
    labels = [f"{m.split('_')[-1]}\n{e:,.0f}" for m, e in zip(b["month"], b["number_of_employees"])]
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_xlabel("Campaign period: masked month and number of employees (low to high)")
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(lambda v, _: P.pct(v))
    P.title(ax, "The model mostly decides which periods are worth calling",
            "Test set, periods with at least 100 customers; bars show take-up (blue = above break-even)")
    return P.save(fig, path)
