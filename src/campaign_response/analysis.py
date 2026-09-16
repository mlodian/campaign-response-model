"""Variable relevance and redundancy, computed on the training split only.

* Discriminating power: Weight of Evidence (WoE) / Information Value (IV),
  univariate ROC-AUC (as Gini) and Kolmogorov-Smirnov for numerics, and
  chi-square with Cramer's V for categoricals.
* Redundancy: Spearman correlation and variance inflation factors (VIF)
  for numerics, Cramer's V between categoricals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

IV_BANDS = [(0.02, "not useful"), (0.1, "weak"), (0.3, "medium"), (0.5, "strong"), (np.inf, "suspicious")]


def iv_strength(iv: float) -> str:
    for upper, name in IV_BANDS:
        if iv < upper:
            return name
    return "suspicious"


def bin_series(s: pd.Series, bins: int = 10) -> pd.Series:
    """Quantile bins for numerics (missing gets its own bin); categories as-is."""
    if s.dtype.kind in "fiub":
        edges = np.unique(np.nanquantile(s.dropna(), np.linspace(0, 1, bins + 1)))
        if len(edges) <= 2:
            b = s.astype(str)
        else:
            b = pd.cut(s, edges, include_lowest=True, duplicates="drop").astype(str)
        return b.where(s.notna(), "missing")
    return s.astype(str).fillna("missing")


def woe_table(x: pd.Series, y: pd.Series, bins: int = 10, smoothing: float = 0.5) -> pd.DataFrame:
    """WoE = ln(share of responders / share of non-responders); positive = above-average response."""
    b = bin_series(x, bins)
    t = pd.crosstab(b, y).reindex(columns=[0, 1], fill_value=0)
    t.columns = ["non_resp", "resp"]
    t["n"] = t.sum(axis=1)
    t["response_rate"] = t["resp"] / t["n"]
    p_resp = (t["resp"] + smoothing) / (t["resp"].sum() + smoothing * len(t))
    p_non = (t["non_resp"] + smoothing) / (t["non_resp"].sum() + smoothing * len(t))
    t["woe"] = np.log(p_resp / p_non)
    t["iv_part"] = (p_resp - p_non) * t["woe"]
    return t.reset_index(names="bin")


def information_values(df: pd.DataFrame, features: list[str], target: str, bins: int = 10) -> pd.DataFrame:
    rows = []
    y = df[target]
    for f in features:
        t = woe_table(df[f], y, bins)
        iv = float(t["iv_part"].sum())
        rec = {"feature": f, "iv": iv, "strength": iv_strength(iv), "bins": len(t)}
        x = df[f]
        if x.dtype.kind in "fiub":
            auc = roc_auc_score(y, x.fillna(x.median()))
            rec["gini"] = abs(2 * auc - 1)
            rec["ks"] = ks_2samp(x[y == 1].dropna(), x[y == 0].dropna()).statistic
        else:
            chi2, p, _, _ = chi2_contingency(pd.crosstab(x, y))
            rec["chi2_p_value"] = p
            rec["cramers_v"] = cramers_v(x, y)
            woe = dict(zip(t["bin"], t["woe"]))
            rec["gini"] = abs(2 * roc_auc_score(y, x.astype(str).map(woe)) - 1)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Bias-corrected Cramer's V (Bergsma, 2013)."""
    t = pd.crosstab(a, b)
    chi2 = chi2_contingency(t, correction=False)[0]
    n = t.values.sum()
    r, k = t.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def cramers_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    m = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            m.loc[a, b] = m.loc[b, a] = cramers_v(df[a], df[b])
    return m


def vif_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = df[cols].astype(float)
    X = (X - X.mean()) / X.std()
    X.insert(0, "const", 1.0)
    return pd.DataFrame({
        "feature": cols,
        "vif": [variance_inflation_factor(X.values, i + 1) for i in range(len(cols))],
    }).sort_values("vif", ascending=False).reset_index(drop=True)


def select_features(iv: pd.DataFrame, spearman: pd.DataFrame, numeric: list[str],
                    min_iv: float, max_corr: float, cramers: pd.DataFrame | None = None,
                    max_cramers: float = 0.8) -> tuple[list[str], pd.DataFrame]:
    """Drop variables with IV below ``min_iv``; within each highly associated group
    (|Spearman| for numerics, Cramer's V for categoricals) keep only the member with
    the highest IV. Returns kept features and a decision log."""
    iv_of = iv.set_index("feature")["iv"]
    decisions, dropped = [], set()
    for f in iv["feature"]:
        if iv_of[f] < min_iv:
            dropped.add(f)
            decisions.append({"feature": f, "decision": "drop", "reason": f"IV {iv_of[f]:.3f} < {min_iv}"})
    ranked = [f for f in iv["feature"] if f in numeric and f not in dropped]
    for i, f in enumerate(ranked):
        if f in dropped:
            continue
        for g in ranked[i + 1:]:
            if g not in dropped and abs(spearman.loc[f, g]) > max_corr:
                dropped.add(g)
                decisions.append({"feature": g, "decision": "drop",
                                  "reason": f"|Spearman| {abs(spearman.loc[f, g]):.2f} with {f} (higher IV)"})
    if cramers is not None:
        cats = [f for f in iv["feature"] if f in cramers.index and f not in dropped]
        for i, f in enumerate(cats):
            if f in dropped:
                continue
            for g in cats[i + 1:]:
                if g not in dropped and cramers.loc[f, g] > max_cramers:
                    dropped.add(g)
                    decisions.append({"feature": g, "decision": "drop",
                                      "reason": f"Cramer's V {cramers.loc[f, g]:.2f} with {f} (higher IV)"})
    kept = [f for f in iv["feature"] if f not in dropped]
    for f in kept:
        decisions.append({"feature": f, "decision": "keep", "reason": f"IV {iv_of[f]:.3f} ({iv_strength(iv_of[f])})"})
    return kept, pd.DataFrame(decisions)
