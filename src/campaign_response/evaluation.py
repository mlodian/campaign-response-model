"""Metrics, profit-based cut-off, stability and result charts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

from . import plotting as P


# ---------------------------------------------------------------- metrics

def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def lift_at(y: np.ndarray, p: np.ndarray, share: float) -> float:
    k = max(1, int(round(share * len(y))))
    top = np.argsort(-p, kind="stable")[:k]
    return float(y[top].mean() / y.mean())


def classification_metrics(y, p, threshold) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    auc = roc_auc_score(y, p)
    return {
        "threshold": threshold, "pr_auc": average_precision_score(y, p), "roc_auc": auc, "gini": 2 * auc - 1,
        "ks": ks_statistic(y, p), "brier": brier_score_loss(y, p), "lift_top10": lift_at(y, p, 0.1),
        "precision": precision_score(y, pred), "recall": recall_score(y, pred), "f1": f1_score(y, pred),
        "specificity": tn / (tn + fp), "accuracy": (tp + tn) / len(y), "call_rate": pred.mean(),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn), "n": int(len(y)), "base_rate": float(y.mean()),
    }


def gains_table(y, p, groups: int = 10) -> pd.DataFrame:
    order = np.argsort(-p, kind="stable")
    ys = y[order]
    decile = np.minimum((np.arange(len(ys)) * groups) // len(ys), groups - 1)
    t = pd.DataFrame({"decile": decile + 1, "y": ys}).groupby("decile")["y"].agg(["size", "sum", "mean"])
    t.columns = ["customers", "responders", "response_rate"]
    t["cumulative_share_of_responders"] = t["responders"].cumsum() / t["responders"].sum()
    t["lift"] = t["response_rate"] / y.mean()
    return t.reset_index()


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index of scores (decile bins set on ``expected``)."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


# ---------------------------------------------------------------- profit

def value_per_sale(campaign: dict) -> float:
    """Expected net profit of a take-up, weighting the risk bands by their share."""
    return float(sum(b["share"] * b["profit_taken_up"] for b in campaign["risk_bands"].values()))


def break_even_probability(campaign: dict) -> float:
    cost = -campaign["cost_not_taken_up"]
    return cost / (value_per_sale(campaign) + cost)


def profit_curve(y, p, campaign) -> pd.DataFrame:
    v, c = value_per_sale(campaign), -campaign["cost_not_taken_up"]
    size = campaign["list_size"]
    rows = []
    for t in np.round(np.arange(0.01, 0.91, 0.01), 2):
        pred = p >= t
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        rows.append({"threshold": t, "call_rate": pred.mean(),
                     "profit_per_list": (tp * v - fp * c) / len(y) * size})
    return pd.DataFrame(rows)


def best_threshold(curve: pd.DataFrame) -> float:
    return float(curve.loc[curve["profit_per_list"].idxmax(), "threshold"])


def alternative_thresholds(y, p) -> dict:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    fpr, tpr, rthr = roc_curve(y, p)
    return {"f1_optimal": float(thr[np.argmax(f1)]), "youden_j": float(rthr[np.argmax(tpr - fpr)]),
            "default_0_5": 0.5, "base_rate": float(y.mean())}


# ---------------------------------------------------------------- charts

def plot_ladder(summary: pd.DataFrame, path: Path) -> Path:
    import matplotlib.pyplot as plt

    s = summary.iloc[::-1].reset_index(drop=True)
    best = summary.loc[summary["pr_auc_mean"].idxmax(), "model"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (col, label, xmax) in zip(axes, [("pr_auc", "PR-AUC (primary)", 0.6), ("gini", "Gini (= 2 x ROC-AUC - 1)", 0.7)]):
        mean = s[f"{col}_mean"] if col != "gini" else s["gini_mean"]
        err = s[f"{col}_std"] if col != "gini" else 2 * s["roc_auc_std"]
        colors = [P.BLUE if m == best else P.CONTEXT for m in s["model"]]
        ax.barh(s.index, mean, xerr=err, height=0.55, color=colors,
                error_kw={"ecolor": P.INK_2, "elinewidth": 1, "capsize": 3})
        for i, (m, e) in enumerate(zip(mean, err)):
            ax.text(m + e + 0.008, i, f"{m:.3f}", va="center", fontsize=10)
        ax.set_yticks(s.index, s["model"])
        ax.tick_params(axis="y", length=0, labelcolor=P.INK_2)
        ax.set_xlim(0, xmax)
        ax.set_title(label, fontsize=12, loc="left")
        ax.grid(axis="y", visible=False)
    if not P.SLIDE_MODE:
        fig.suptitle(f"{best} ranks responders best in 5-fold cross-validation", x=0.01, ha="left",
                     fontsize=14, fontweight="bold")
    return P.save(fig, path)


def plot_overfit(summary: pd.DataFrame, path: Path) -> Path:
    fig, ax = P.new_figure(9, 4.4, grid_axis="x")
    s = summary[~summary["model"].str.startswith(("Baseline", "Business"))].iloc[::-1].reset_index(drop=True)
    y = np.arange(len(s))
    ax.barh(y + 0.17, s["train_pr_auc_mean"], height=0.32, color=P.CONTEXT, label="Training folds")
    ax.barh(y - 0.17, s["pr_auc_mean"], height=0.32, color=P.BLUE, label="Validation folds")
    for i, (tr, va) in enumerate(zip(s["train_pr_auc_mean"], s["pr_auc_mean"])):
        ax.text(max(tr, va) + 0.01, i, f"gap {tr - va:+.3f}", va="center", fontsize=9.5, color=P.INK_2)
    ax.set_yticks(y, s["model"])
    ax.tick_params(axis="y", length=0, labelcolor=P.INK_2)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("PR-AUC")
    ax.legend(loc="lower right")
    lg = summary.set_index("model")["overfit_gap_pr_auc"]
    headline = (f"Tuning cut LightGBM's train-validation gap from {lg['LightGBM']:.2f} to {lg['LightGBM (tuned)']:.2f}"
                if "LightGBM (tuned)" in lg else "Train-validation gaps by model")
    P.title(ax, headline, "Mean PR-AUC on training vs validation folds; a large gap signals overfitting")
    return P.save(fig, path)


def plot_learning_curve(lc: pd.DataFrame, path: Path) -> Path:
    fig, ax = P.new_figure(8, 4.4)
    for col, color, lab in [("train", P.CONTEXT, "Training"), ("valid", P.BLUE, "Validation")]:
        ax.plot(lc["train_size"], lc[f"{col}_mean"], color=color, marker="o", ms=6, mec=P.SURFACE, mew=2, label=lab)
        ax.fill_between(lc["train_size"], lc[f"{col}_mean"] - lc[f"{col}_std"],
                        lc[f"{col}_mean"] + lc[f"{col}_std"], color=color, alpha=0.12, lw=0)
    ax.set(xlabel="Training rows", ylabel="PR-AUC")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    ax.legend(loc="lower right")
    P.title(ax, "Validation score plateaus and the gap narrows as data grows", "Learning curve, 5-fold CV")
    return P.save(fig, path)


def plot_boosting_curve(bc: pd.DataFrame, path: Path) -> Path:
    fig, ax = P.new_figure(8, 4.2)
    ax.plot(bc["round"], bc["train_logloss"], color=P.CONTEXT, label="Training")
    ax.plot(bc["round"], bc["valid_logloss"], color=P.BLUE, label="Validation")
    best = int(bc.loc[bc["valid_logloss"].idxmin(), "round"])
    ax.axvline(best, color=P.MUTED, lw=1)
    ax.text(best, bc["valid_logloss"].min() + 0.012, f"lowest validation loss\nat round {best}  ", color=P.MUTED,
            fontsize=9, ha="right")
    ax.set(xlabel="Boosting round", ylabel="Log-loss")
    ax.legend(loc="upper center")
    P.title(ax, "Validation loss flattens instead of rising: no late overfitting",
            "Tuned LightGBM, inner 80/20 split of the training data")
    return P.save(fig, path)


def plot_calibration(y, p, path: Path) -> Path:
    frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = P.new_figure(5.6, 5)
    top = max(frac.max(), mean.max()) * 1.08
    ax.plot([0, top], [0, top], color=P.MUTED, lw=1)
    ax.plot(mean, frac, color=P.BLUE, marker="o", ms=7, mec=P.SURFACE, mew=2)
    ax.set(xlabel="Predicted take-up probability", ylabel="Observed take-up rate", xlim=(0, top), ylim=(0, top))
    P.title(ax, "Predicted probabilities match observed take-up", f"Test deciles; Brier {brier_score_loss(y, p):.3f}")
    return P.save(fig, path)


def plot_gains(g: pd.DataFrame, path: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    x = np.r_[0, g["decile"] / 10]
    ax.plot(x, np.r_[0, g["cumulative_share_of_responders"]], color=P.BLUE, marker="o", ms=6, mec=P.SURFACE, mew=2,
            label="Model")
    ax.plot([0, 1], [0, 1], color=P.MUTED, lw=1, label="Random calling")
    ax.set(xlabel="Share of list called (highest score first)", ylabel="Share of all takers reached",
           xlim=(0, 1), ylim=(0, 1.02))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(loc="lower right")
    ax.set_title("Cumulative gains", fontsize=12, loc="left")
    ax = axes[1]
    colors = [P.BLUE if d <= 2 else P.CONTEXT for d in g["decile"]]
    ax.bar(g["decile"], g["lift"], color=colors, width=0.6)
    for d, l in zip(g["decile"], g["lift"]):
        ax.text(d, l + 0.05, f"{l:.1f}x", ha="center", fontsize=9.5, color=P.INK_2)
    ax.axhline(1, color=P.MUTED, lw=1)
    ax.set(xlabel="Score decile (1 = highest)", ylabel="Lift over average take-up")
    ax.set_xticks(g["decile"])
    ax.set_title("Lift by decile", fontsize=12, loc="left")
    ax.grid(axis="x", visible=False)
    if not P.SLIDE_MODE:
        fig.suptitle(f"The top decile converts at {g['lift'].iloc[0]:.1f}x the average; the top 30% "
                     f"holds {g['cumulative_share_of_responders'].iloc[2]:.0%} of takers",
                     x=0.01, ha="left", fontsize=14, fontweight="bold")
    return P.save(fig, path)


def plot_ks(y, p, path: Path) -> Path:
    order = np.argsort(-p, kind="stable")
    ys = y[order]
    x = np.arange(1, len(ys) + 1) / len(ys)
    cum1 = np.cumsum(ys) / ys.sum()
    cum0 = np.cumsum(1 - ys) / (1 - ys).sum()
    i = int(np.argmax(cum1 - cum0))
    fig, ax = P.new_figure(8, 4.4)
    ax.plot(x, cum1, color=P.BLUE, label="Takers")
    ax.plot(x, cum0, color=P.ORANGE, label="Non-takers")
    ax.vlines(x[i], cum0[i], cum1[i], color=P.INK_2, lw=1)
    ax.text(x[i] + 0.01, (cum0[i] + cum1[i]) / 2, f"KS = {cum1[i] - cum0[i]:.2f}", fontsize=10)
    ax.set(xlabel="Share of list (highest score first)", ylabel="Cumulative share", xlim=(0, 1), ylim=(0, 1.02))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(loc="lower right")
    P.title(ax, "Takers and non-takers separate early in the ranking", "Kolmogorov-Smirnov, test set")
    return P.save(fig, path)


def plot_profit(curve: pd.DataFrame, chosen: float, break_even: float, others: dict, path: Path) -> Path:
    fig, ax = P.new_figure(9, 4.6)
    ax.plot(curve["threshold"], curve["profit_per_list"] / 1e3, color=P.BLUE)
    best = curve.loc[curve["threshold"] == chosen].iloc[0]
    ax.scatter([chosen], [best["profit_per_list"] / 1e3], s=70, color=P.BLUE, edgecolor=P.SURFACE, lw=2, zorder=3)
    ax.annotate(f"cut-off {chosen:.2f}: {best['profit_per_list'] / 1e3:,.0f}k per list,\ncalling {best['call_rate']:.0%} "
                f"of customers (out-of-fold)", (chosen, best["profit_per_list"] / 1e3), xytext=(60, -70),
                textcoords="offset points", fontsize=10, arrowprops={"arrowstyle": "-", "color": P.MUTED})
    ax.axvline(break_even, color=P.MUTED, lw=1)
    ax.text(break_even, ax.get_ylim()[0], f" break-even {break_even:.2f}", color=P.MUTED, fontsize=9, va="bottom")
    for name, t in [("0.5 default", others["default_0_5"])]:
        v = curve.iloc[(curve["threshold"] - t).abs().argmin()]["profit_per_list"] / 1e3
        ax.scatter([t], [v], s=40, color=P.CONTEXT, edgecolor=P.INK_2, lw=0.8, zorder=3)
        ax.annotate(name, (t, v), xytext=(6, -14), textcoords="offset points", fontsize=9, color=P.INK_2)
    ax.axhline(0, color=P.AXIS, lw=1)
    ax.set(xlabel="Cut-off (predicted take-up probability)", ylabel="Net profit per 10,000-customer list (k)")
    P.title(ax, f"A {chosen:.2f} cut-off maximises campaign profit on out-of-fold scores",
            "Profit per sale = risk-weighted Finance estimate; unsuccessful call = -300")
    return P.save(fig, path)


def plot_confusion(m: dict, path: Path) -> Path:
    fig, ax = P.new_figure(6.2, 5)
    ax.grid(False)
    cells = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    names = np.array([["True negative\nnot called, would not buy", "False positive\ncalled, did not buy"],
                      ["False negative\nnot called, would have bought", "True positive\ncalled and bought"]])
    shade = np.array([[P.CONTEXT, "#f5d3c4"], ["#f5d3c4", "#cde2fb"]])
    for i in range(2):
        for j in range(2):
            ax.add_patch(Rectangle((j, 1 - i), 0.97, 0.97, color=shade[i][j]))
            ax.text(j + 0.485, 1 - i + 0.6, f"{cells[i, j]:,}", ha="center", fontsize=18, fontweight="bold")
            ax.text(j + 0.485, 1 - i + 0.43, f"{cells[i, j] / cells.sum():.1%}", ha="center", fontsize=10,
                    color=P.INK_2)
            ax.text(j + 0.485, 1 - i + 0.12, names[i][j], ha="center", fontsize=8.5, color=P.INK_2)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.485, 1.485], ["Predicted: don't call", "Predicted: call"])
    ax.set_yticks([1.485, 0.485], ["Did not take up", "Took up"])
    ax.tick_params(length=0, labelcolor=P.INK_2)
    for s in ax.spines.values():
        s.set_visible(False)
    P.title(ax, f"Precision {m['precision']:.0%}, recall {m['recall']:.0%} at cut-off {m['threshold']:.2f}",
            f"Held-out test set, {m['n']:,} customers")
    return P.save(fig, path)


def plot_psi(train_p, test_p, value: float, path: Path) -> Path:
    fig, ax = P.new_figure(8, 4.2)
    bins = np.linspace(0, 1, 41)
    ax.hist(train_p, bins=bins, density=True, color=P.CONTEXT, label="Training (out-of-fold)")
    ax.hist(test_p, bins=bins, density=True, histtype="step", color=P.BLUE, lw=2, label="Test")
    ax.set(xlabel="Predicted take-up probability", ylabel="Density")
    ax.legend(loc="upper right")
    P.title(ax, f"Score distributions are stable between samples (PSI {value:.3f})",
            "By a common rule of thumb, PSI below 0.10 indicates little change")
    return P.save(fig, path)
