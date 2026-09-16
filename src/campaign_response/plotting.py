"""Shared chart style: one accent colour, muted chrome, thin marks, direct labels."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Slide mode drops chart titles so the slide headline carries the takeaway.
SLIDE_MODE = os.environ.get("MR_SLIDE_FIGURES") == "1"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CONTEXT = "#d3d1c7"   # de-emphasised marks
BLUE = "#2a78d6"      # series 1: approved / the model / the point being made
ORANGE = "#eb6834"    # series 2: denied
AQUA = "#1baf7a"      # series 3

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 14, "axes.labelsize": 10.5,
    "axes.labelcolor": INK_2, "text.color": INK, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "legend.frameon": False,
    "legend.fontsize": 10, "figure.dpi": 110, "savefig.dpi": 200,
})


def new_figure(width: float = 9, height: float = 5, grid_axis: str = "y"):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.grid(axis="x" if grid_axis == "x" else "y")
    ax.grid(axis="y" if grid_axis == "x" else "x", visible=False)
    return fig, ax


def title(ax, text: str, subtitle: str | None = None) -> None:
    """Takeaway as the title; the subtitle says what is plotted."""
    if SLIDE_MODE:
        return
    ax.set_title(text, loc="left", pad=28 if subtitle else 14)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_2, fontsize=10.5, va="bottom")


def bars(ax, labels, values, colors=None, horizontal=True, fmt=None, thickness=0.6, pad=None):
    """Thin bars from a single baseline, value labelled at the tip."""
    colors = colors or [BLUE] * len(values)
    pos = range(len(values))
    if horizontal:
        ax.barh(pos, values, height=thickness, color=colors, linewidth=0)
        ax.set_yticks(list(pos), labels)
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0, labelcolor=INK_2)
        span = max(values)
        pad = span * 0.012 if pad is None else pad
        for p, v in zip(pos, values):
            if fmt:
                ax.text(v + pad, p, fmt(v), va="center", fontsize=9.5, color=INK_2)
        ax.set_xlim(0, span * 1.15)
    else:
        ax.bar(pos, values, width=thickness, color=colors, linewidth=0)
        ax.set_xticks(list(pos), labels)
        ax.tick_params(axis="x", length=0, labelcolor=INK_2)
        span = max(values)
        for p, v in zip(pos, values):
            if fmt:
                ax.text(p, v + span * 0.015, fmt(v), ha="center", fontsize=9.5, color=INK_2)
        ax.set_ylim(0, span * 1.15)


def save(fig, path: Path) -> Path:
    if SLIDE_MODE:
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * 0.74, h * 0.74)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def pct(v: float, digits: int = 0) -> str:
    return f"{v * 100:.{digits}f}%"
