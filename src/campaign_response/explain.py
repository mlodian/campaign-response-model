"""Model interpretation: SHAP, permutation importance and scorecard coefficients."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from . import plotting as P
from .data import label


def pretty(name: str, raw_features: list[str]) -> str:
    """'employment_type_Cat_1_b***r' -> 'Employment type: Cat_1_b***r'."""
    for raw in sorted(raw_features, key=len, reverse=True):
        if name.startswith(raw + "_") and name != raw:
            return f"{label(raw)}: {name[len(raw) + 1:]}"
    return label(name)


def shap_values(pipe: Pipeline, X: pd.DataFrame, n: int, seed: int, raw_features: list[str]):
    import shap

    sample = X.sample(min(n, len(X)), random_state=seed)
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    Xt = prep.transform(sample)
    cols = prep[-1].get_feature_names_out()
    names = [pretty(c, raw_features) for c in cols]
    Xt = pd.DataFrame(np.asarray(Xt, dtype=float), columns=names)
    if hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, Xt)
        values = explainer.shap_values(Xt)
    else:
        values = shap.TreeExplainer(model).shap_values(Xt)
        if isinstance(values, list):
            values = values[1]
        if values.ndim == 3:
            values = values[:, :, 1]
    return values, Xt


def shap_by_feature(values, Xt: pd.DataFrame, raw_features: list[str]) -> pd.DataFrame:
    """Mean |SHAP| per encoded column, and summed back to the original variable."""
    per_col = pd.Series(np.abs(values).mean(0), index=Xt.columns)
    rows = []
    for raw in raw_features:
        lab = label(raw)
        cols = [c for c in per_col.index if c == lab or c.startswith(lab + ":")]
        rows.append({"feature": raw, "mean_abs_shap": float(per_col[cols].sum())})
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def permutation_table(pipe, X, y, seed) -> pd.DataFrame:
    r = permutation_importance(pipe, X, y, scoring="average_precision", n_repeats=5, random_state=seed, n_jobs=1)
    return (pd.DataFrame({"feature": X.columns, "pr_auc_drop": r.importances_mean, "std": r.importances_std})
            .sort_values("pr_auc_drop", ascending=False).reset_index(drop=True))


def plot_importance(shap_t: pd.DataFrame, perm: pd.DataFrame, path: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, t, col, xl, ttl in [
        (axes[0], shap_t, "mean_abs_shap", "Mean |SHAP| (log-odds)", "SHAP (model's own view)"),
        (axes[1], perm, "pr_auc_drop", "Drop in test PR-AUC when shuffled", "Permutation (effect on test score)"),
    ]:
        t = t.sort_values(col, ascending=False).reset_index(drop=True)
        colors = [P.BLUE] * 3 + [P.CONTEXT] * (len(t) - 3)
        P.bars(ax, [label(f) for f in t["feature"]], list(t[col].clip(lower=0)), colors=colors[:len(t)],
               fmt=lambda v: f"{v:.3f}", thickness=0.6)
        ax.set_xlabel(xl)
        ax.set_title(ttl, fontsize=12, loc="left")
        ax.grid(axis="y", visible=False)
    if not P.SLIDE_MODE:
        fig.suptitle("Macro conditions, previous-campaign history and contact medium matter most",
                     x=0.01, ha="left", fontsize=14, fontweight="bold")
    return P.save(fig, path)


def plot_beeswarm(values, Xt, path: Path, top: int = 12) -> Path:
    import matplotlib.pyplot as plt
    import shap
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("bo", [P.BLUE, "#e6e4dd", P.ORANGE])
    shap.summary_plot(values, Xt, max_display=top, show=False, cmap=cmap, plot_size=(9, 6))
    fig = plt.gcf()
    if not P.SLIDE_MODE:
        fig.axes[0].set_title("A low employment indicator, mobile contact and recent prior contact raise take-up", loc="left", fontsize=14,
                              fontweight="bold", pad=14)
    fig.axes[0].set_xlabel("SHAP value (impact on log-odds of take-up)")
    return P.save(fig, path)


def scorecard_table(pipe: Pipeline) -> pd.DataFrame:
    """Coefficient per WoE variable for the logistic scorecard (positive = more take-up)."""
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    names = prep[-1].get_feature_names_out()
    return (pd.DataFrame({"feature": names, "coefficient": model.coef_[0]})
            .sort_values("coefficient", key=np.abs, ascending=False).reset_index(drop=True))
