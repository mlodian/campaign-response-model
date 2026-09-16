"""Audit and clean the campaign file.

Every step records what it changed in an ``AuditLog`` so the preparation can be
reviewed line by line. Anything that has to be *learned* from data (caps, rare
category groups, bins) is fitted later on the training split only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import CATEGORICAL_RAW, NUMERIC_RAW

TARGET = "response"
NEVER_CONTACTED = 999

# Masked category labels keep their first and last letters. "u***n" is read as
# "unknown", i.e. a missing value coded as its own category.
UNKNOWN_PATTERN = r"_u\*\*\*n$"


@dataclass
class AuditLog:
    steps: list[dict] = field(default_factory=list)

    def add(self, step: str, rows: int, affected: int, note: str) -> None:
        self.steps.append({"step": step, "rows": rows, "affected": affected, "note": note})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: type, missing/unknown share, spread and response contrast."""
    rows = []
    for col in NUMERIC_RAW + CATEGORICAL_RAW:
        s = df[col]
        rec = {"column": col, "kind": "numeric" if col in NUMERIC_RAW else "categorical",
               "distinct": s.nunique(), "nulls": int(s.isna().sum())}
        if col in NUMERIC_RAW:
            q = s.quantile([0, 0.01, 0.5, 0.99, 1])
            rec.update(min=q[0], p01=q[0.01], median=q[0.5], p99=q[0.99], max=q[1],
                       above_p99=int((s > q[0.99]).sum()),
                       sentinel_999=int((s == NEVER_CONTACTED).sum()) if col == "days_since_prev_campaign" else 0)
        else:
            rec.update(unknown_share=float(s.str.contains(UNKNOWN_PATTERN).mean()),
                       smallest_level_share=float(s.value_counts(normalize=True).min()))
        rows.append(rec)
    return pd.DataFrame(rows)


def clean(df: pd.DataFrame, leakage_columns: list[str]) -> tuple[pd.DataFrame, AuditLog]:
    log = AuditLog()
    log.add("raw records", len(df), 0, f"{df.shape[1]} columns, {df.isna().sum().sum()} nulls")

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log.add("drop exact duplicates", len(df), before - len(df), "identical across all 21 fields")

    out = df.copy()
    out[TARGET] = out[TARGET].astype(int)

    # 999 is a code for "never contacted", not a number of days.
    never = out["days_since_prev_campaign"] == NEVER_CONTACTED
    out["previously_contacted"] = (~never).astype(int)
    out["prev_contact_recency"] = pd.cut(
        out["days_since_prev_campaign"].where(~never), [-1, 3, 6, 14, 998],
        labels=["0-3 days", "4-6 days", "7-14 days", "15+ days"],
    ).cat.add_categories("never").fillna("never").astype(str)
    log.add("decode 999 sentinel", len(out), int(never.sum()),
            "days_since_prev_campaign = 999 -> 'never' recency band + previously_contacted flag")

    inconsistent = (~never) & (out["previous_outcome"].str.endswith("_n***t"))
    log.add("check sentinel vs previous outcome", len(out), int(inconsistent.sum()),
            "'nonexistent' previous outcome must coincide with 999 (0 conflicts expected)")

    unknown = sum(int(out[c].str.contains(UNKNOWN_PATTERN).sum()) for c in CATEGORICAL_RAW)
    log.add("keep 'unknown' as a category", len(out), unknown,
            "unknown is informative (e.g. credit-in-default unknown converts at less than half the rate)")

    zero_calls = int((out["call_duration_s"] == 0).sum())
    log.add("leakage: call duration", len(out), zero_calls,
            f"known only after the call; {zero_calls} zero-second calls all have response 0; excluded from features")

    extreme = int((out["call_attempts"] > out["call_attempts"].quantile(0.99)).sum())
    log.add("flag extreme call attempts", len(out), extreme,
            "above the 99th percentile; capped inside the model pipeline (cap learned on training data)")

    for col in leakage_columns:
        out = out.rename(columns={col: f"leak_{col}"})
    log.add("final modelling table", len(out), out.shape[1], "rows / columns")
    return out, log
