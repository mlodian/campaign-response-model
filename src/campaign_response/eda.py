"""Exploratory and variable-analysis charts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import plotting as P
from .analysis import woe_table
from .data import label

STRENGTH_COLOR = {"not useful": "#e1e0d9", "weak": P.CONTEXT, "medium": "#86b6ef", "strong": P.BLUE,
                  "suspicious": "#1c5cab"}


def plot_leakage(df: pd.DataFrame, path: Path) -> Path:
    d = df.assign(decile=pd.qcut(df["leak_call_duration_s"], 10, labels=False) + 1)
    g = d.groupby("decile").agg(rate=("response", "mean"), lo=("leak_call_duration_s", "min"),
                                hi=("leak_call_duration_s", "max")).reset_index()
    fig, ax = P.new_figure(9.5, 4.5)
    colors = [P.ORANGE if r > df["response"].mean() else P.CONTEXT for r in g["rate"]]
    P.bars(ax, [f"{int(a)}-{int(b)}s" for a, b in zip(g["lo"], g["hi"])], list(g["rate"]), colors=colors,
           horizontal=False, fmt=lambda v: f"{v:.0%}")
    ax.axhline(df["response"].mean(), color=P.MUTED, lw=1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_ylabel("Take-up rate")
    P.title(ax, "Call duration is the strongest signal, but it is only known once the call has ended",
            "Take-up by call-duration decile; excluded from the model (data dictionary: known only after the call)")
    return P.save(fig, path)


def plot_outliers(df: pd.DataFrame, path: Path, cap_q: float) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col in zip(axes, ["call_attempts", "age"]):
        s = df[col]
        cap = s.quantile(cap_q)
        vals = s.value_counts().sort_index()
        ax.bar(vals.index, vals.values, width=0.8, color=[P.ORANGE if v > cap else P.BLUE for v in vals.index])
        ax.axvline(cap, color=P.MUTED, lw=1)
        ax.text(cap, vals.max() * 0.95, f"  99th pct = {cap:.0f}\n  {int((s > cap).sum()):,} rows above",
                fontsize=9, color=P.INK_2, va="top")
        ax.set_yscale("log")
        ax.set_xlabel(label(col))
        ax.set_ylabel("Customers (log scale)")
        ax.set_title(label(col), fontsize=12, loc="left")
    if not P.SLIDE_MODE:
        fig.suptitle("Call attempts has a long tail and is capped; age extremes are plausible and kept",
                     x=0.01, ha="left", fontsize=14, fontweight="bold")
    return P.save(fig, path)


def plot_iv(iv: pd.DataFrame, path: Path) -> Path:
    t = iv.sort_values("iv", ascending=False).reset_index(drop=True)
    fig, ax = P.new_figure(9.5, 6.2, grid_axis="x")
    names = [label(f.replace("leak_", "")) + (" (leak)" if f.startswith("leak_") else "") for f in t["feature"]]
    colors = [P.ORANGE if f.startswith("leak_") else STRENGTH_COLOR[s] for f, s in zip(t["feature"], t["strength"])]
    P.bars(ax, names, list(t["iv"]), colors=colors, fmt=lambda v: f"{v:.3f}", thickness=0.62)
    for x, lab in [(0.02, "0.02"), (0.1, "0.1"), (0.3, "0.3"), (0.5, "0.5")]:
        ax.axvline(x, color=P.GRID, lw=1, zorder=0)
    ax.set_xlabel("Information Value (training split)")
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=STRENGTH_COLOR[k], label=k) for k in STRENGTH_COLOR] +
              [Patch(color=P.ORANGE, label="post-call leak")], loc="lower right", title="IV strength", fontsize=9)
    P.title(ax, "Weekday and the two loan flags carry almost no information",
            "IV bands: <0.02 not useful, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong, >0.5 suspicious")
    return P.save(fig, path)


def plot_heatmap(m: pd.DataFrame, path: Path, title: str, subtitle: str, diverging: bool) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "div", ["#d03b3b", "#f0efec", P.BLUE]) if diverging else LinearSegmentedColormap.from_list(
        "seq", ["#f7f7f4", "#9ec5f4", "#184f95"])
    n = len(m)
    fig, ax = plt.subplots(figsize=(0.62 * n + 3.2, 0.52 * n + 1.6))
    im = ax.imshow(m.values, cmap=cmap, vmin=-1 if diverging else 0, vmax=1)
    labels = [label(c) for c in m.columns]
    ax.set_xticks(range(n), labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(n), labels, fontsize=9)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            v = m.values[i, j]
            if i != j:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.6 else P.INK)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    P.title(ax, title, subtitle)
    return P.save(fig, path)


def plot_woe_curve(df: pd.DataFrame, feature: str, path: Path, title: str) -> Path:
    t = woe_table(df[feature], df["response"])
    t["lo"] = t["bin"].str.extract(r"[\(\[]\s*([-\d.]+)").astype(float)
    t = t.sort_values("lo")
    fig, ax = P.new_figure(8.5, 4.2)
    colors = [P.BLUE if w > 0 else P.CONTEXT for w in t["woe"]]
    lo_hi = t["bin"].str.extract(r"[\(\[]\s*([-\d.]+),\s*([-\d.]+)").astype(float)
    names = [f"{int(np.floor(a)) + 1}-{int(b)}" for a, b in zip(lo_hi[0], lo_hi[1])]
    P.bars(ax, names, list(t["woe"]), colors=colors, horizontal=False)
    ax.set_ylim(t["woe"].min() * 1.25, t["woe"].max() * 1.25)
    ax.axhline(0, color=P.MUTED, lw=1)
    for i, (w, r) in enumerate(zip(t["woe"], t["response_rate"])):
        ax.text(i, w + (0.03 if w >= 0 else -0.08), f"{r:.0%}", ha="center", fontsize=9, color=P.INK_2)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_ylabel("Weight of Evidence")
    P.title(ax, title, f"{label(feature)} deciles; labels show take-up rate")
    return P.save(fig, path)


def plot_rate(df: pd.DataFrame, col: str, path: Path, title: str, subtitle: str, order=None,
              highlight: set | None = None, width=9, height=4.4) -> Path:
    g = df.groupby(col)["response"].agg(["mean", "size"])
    g = g.reindex(order) if order else g.sort_values("mean", ascending=False)
    base = df["response"].mean()
    fig, ax = P.new_figure(width, height, grid_axis="x")
    colors = [P.BLUE if (highlight and k in highlight) else P.CONTEXT for k in g.index]
    P.bars(ax, [f"{k}  (n={n:,})" for k, n in zip(g.index, g["size"])], list(g["mean"]), colors=colors,
           fmt=lambda v: f"{v:.0%}")
    ax.axvline(base, color=P.MUTED, lw=1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Take-up rate")
    P.title(ax, title, f"{subtitle}; grey line = overall {base:.1%}")
    return P.save(fig, path)


def plot_macro(df: pd.DataFrame, path: Path) -> Path:
    g = df.groupby("number_of_employees")["response"].agg(["mean", "size"]).reset_index()
    fig, ax = P.new_figure(8.5, 4.2)
    ax.scatter(g["number_of_employees"], g["mean"], s=np.sqrt(g["size"]) * 3, color=P.BLUE, alpha=0.8,
               edgecolor=P.SURFACE, lw=1.5)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set(xlabel="Number of employees (macro indicator)", ylabel="Take-up rate")
    P.title(ax, "Take-up is several times higher when the employment indicator is low",
            "Each dot is one value of the quarterly indicator; size = customers contacted")
    return P.save(fig, path)
