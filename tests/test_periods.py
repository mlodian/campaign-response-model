import numpy as np
import pandas as pd
import pytest

from campaign_response import periods as per
from campaign_response.modeling import paired_differences


def test_period_ids_group_identical_macro_rows():
    df = pd.DataFrame({"employment_variation_rate": [1.1, 1.1, -1.8], "consumer_price_index": [93.9, 93.9, 92.9],
                       "consumer_confidence_index": [-36.4, -36.4, -46.2], "number_of_employees": [5191, 5191, 5099]})
    ids = per.period_ids(df)
    assert ids[0] == ids[1] != ids[2]


def test_within_period_ignores_between_period_signal():
    # score = period take-up rate: perfect pooled ranking, no ranking inside a period
    y = np.concatenate([np.tile([1, 1, 1] + [0] * 7, 10), np.tile([1] + [0] * 9, 10)])  # ties keep row order
    period = np.array([0] * 100 + [1] * 100)
    p = np.where(period == 0, 0.3, 0.1)
    out = per.within_period(y, p, period)
    assert out["periods_scored"] == 2
    assert out["within_roc_auc"] == pytest.approx(0.5)
    assert out["within_lift_top10"] == pytest.approx(1.0)


def test_paired_differences_uses_same_folds():
    folds = pd.DataFrame({"model": ["A"] * 3 + ["B"] * 3, "fold": [0, 1, 2] * 2,
                          "pr_auc": [0.5, 0.4, 0.6, 0.45, 0.42, 0.5]})
    d = paired_differences(folds, "A").iloc[0]
    assert d["model"] == "B"
    assert d["pr_auc_diff_mean"] == pytest.approx((0.05 - 0.02 + 0.1) / 3)
    assert d["reference_wins"] == 2
