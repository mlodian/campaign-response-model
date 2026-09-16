"""Transformers and preprocessing pipelines.

Two encodings are used on purpose:

* **WoE encoding** (binned numerics, grouped categories) for the logistic
  scorecard. Binning lets a linear model capture non-monotonic effects such as
  age, and WoE puts every input on the same log-odds scale.
* **One-hot encoding** with capped, scaled numerics for the other logistic model,
  and unscaled numerics for tree models, which need neither binning nor scaling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COUNT_FEATURES = ["call_attempts", "prev_campaign_attempts"]
RARE_LABEL = "Other (rare)"


class QuantileCapper(BaseEstimator, TransformerMixin):
    """Cap selected columns at an upper quantile learned during ``fit``."""

    def __init__(self, columns=(), quantile: float = 0.99):
        self.columns = columns
        self.quantile = quantile

    def fit(self, X, y=None):
        self.caps_ = {c: float(X[c].quantile(self.quantile)) for c in self.columns if c in X}
        return self

    def transform(self, X):
        X = X.copy()
        for c, cap in self.caps_.items():
            X[c] = X[c].clip(upper=cap)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class RareGrouper(BaseEstimator, TransformerMixin):
    """Merge categories rarer than ``min_share`` (learned during ``fit``) into one level."""

    def __init__(self, columns=(), min_share: float = 0.01):
        self.columns = columns
        self.min_share = min_share

    def fit(self, X, y=None):
        self.keep_ = {c: set(X[c].value_counts(normalize=True).loc[lambda s: s >= self.min_share].index)
                      for c in self.columns if c in X}
        return self

    def transform(self, X):
        X = X.copy()
        for c, keep in self.keep_.items():
            X[c] = X[c].astype(str).where(X[c].astype(str).isin(keep), RARE_LABEL)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class WoEEncoder(BaseEstimator, TransformerMixin):
    """Replace each value with the Weight of Evidence of its bin.

    Numerics are cut at training-set quantiles; categories are used as-is. WoE is
    ln(share of responders / share of non-responders) with additive smoothing, so
    unseen or empty bins map close to 0 (no evidence either way).
    """

    def __init__(self, numeric=(), categorical=(), bins: int = 10, smoothing: float = 0.5):
        self.numeric = numeric
        self.categorical = categorical
        self.bins = bins
        self.smoothing = smoothing

    def _bin(self, col, s):
        if col in self.edges_:
            return pd.cut(s, self.edges_[col], include_lowest=True).astype(str)
        return s.astype(str)

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.edges_, self.maps_ = {}, {}
        for col in list(self.numeric) + list(self.categorical):
            s = X[col]
            if col in self.numeric:
                edges = np.unique(np.quantile(s, np.linspace(0, 1, self.bins + 1)))
                edges[0], edges[-1] = -np.inf, np.inf
                self.edges_[col] = edges
            b = self._bin(col, s)
            t = pd.crosstab(b, y).reindex(columns=[0, 1], fill_value=0)
            k = len(t)
            p1 = (t[1] + self.smoothing) / (t[1].sum() + self.smoothing * k)
            p0 = (t[0] + self.smoothing) / (t[0].sum() + self.smoothing * k)
            self.maps_[col] = np.log(p1 / p0).to_dict()
        self.feature_names_ = list(self.numeric) + list(self.categorical)
        return self

    def transform(self, X):
        out = pd.DataFrame(index=X.index)
        for col in self.feature_names_:
            out[col] = self._bin(col, X[col]).map(self.maps_[col]).astype(float).fillna(0.0)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_, dtype=object)


def split_types(features: list[str], numeric_all: list[str]) -> tuple[list[str], list[str]]:
    return [f for f in features if f in numeric_all], [f for f in features if f not in numeric_all]


def woe_preprocessor(numeric, categorical, rare_share, cap_q):
    return make_pipeline(
        QuantileCapper([c for c in COUNT_FEATURES if c in numeric], cap_q),
        RareGrouper(categorical, rare_share),
        WoEEncoder(numeric, categorical),
    )


def onehot_preprocessor(numeric, categorical, rare_share, cap_q, scale: bool):
    encode = ColumnTransformer(
        [("num", StandardScaler() if scale else "passthrough", list(numeric)),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(categorical))],
        verbose_feature_names_out=False,
    )
    return make_pipeline(
        QuantileCapper([c for c in COUNT_FEATURES if c in numeric], cap_q),
        RareGrouper(categorical, rare_share),
        encode,
    )
