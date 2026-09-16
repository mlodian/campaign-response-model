import numpy as np
import pandas as pd
import pytest

from campaign_response import analysis as A
from campaign_response.cleaning import clean
from campaign_response.data import COLUMN_NAMES
from campaign_response.features import QuantileCapper, RareGrouper, WoEEncoder


def raw_frame(n=6, **overrides):
    base = {
        "age": [30, 45, 60, 25, 38, 52][:n], "call_duration_s": [0, 120, 300, 50, 400, 90][:n],
        "call_attempts": [1, 2, 3, 1, 40, 2][:n], "days_since_prev_campaign": [999, 5, 999, 999, 2, 999][:n],
        "prev_campaign_attempts": [0, 1, 0, 0, 2, 0][:n], "employment_variation_rate": [1.1] * n,
        "consumer_price_index": [93.9] * n, "consumer_confidence_index": [-36.4] * n,
        "interbank_rate_3m": [4.8] * n, "number_of_employees": [5191.0] * n,
        "employment_type": ["Cat_0_a***."] * n, "civil_status": ["Cat_1_m***d"] * n,
        "education": ["Cat_7_u***n"] * n, "credit_in_default": ["Cat_0_n***o"] * n,
        "has_home_loan": ["Cat_0_n***o"] * n, "has_personal_loan": ["Cat_0_n***o"] * n,
        "contact_medium": ["Cat_0_c***r"] * n, "last_contact_month": ["Cat_6_m***y"] * n,
        "last_contact_weekday": ["Cat_1_m***n"] * n,
        "previous_outcome": ["Cat_1_n***t", "Cat_2_s***s", "Cat_1_n***t", "Cat_1_n***t", "Cat_0_f***e",
                             "Cat_1_n***t"][:n],
        "response": [0, 1, 0, 0, 1, 0][:n],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_column_mapping_covers_all_21_fields():
    assert len(COLUMN_NAMES) == 21
    assert COLUMN_NAMES["Feature_dn_1"] == "call_duration_s"


def test_clean_decodes_sentinel_and_isolates_leak():
    df, log = clean(pd.concat([raw_frame(), raw_frame().iloc[[0]]]), ["call_duration_s"])
    assert len(df) == 6  # duplicate removed
    assert "call_duration_s" not in df and "leak_call_duration_s" in df
    assert df["prev_contact_recency"].tolist() == ["never", "4-6 days", "never", "never", "0-3 days", "never"]
    assert df["previously_contacted"].tolist() == [0, 1, 0, 0, 1, 0]
    steps = log.to_frame().set_index("step")["affected"]
    assert steps["check sentinel vs previous outcome"] == 0
    assert steps["drop exact duplicates"] == 1


def test_woe_and_iv_known_values():
    x = pd.Series(["a"] * 50 + ["b"] * 50)
    y = pd.Series([1] * 40 + [0] * 10 + [1] * 10 + [0] * 40)
    t = A.woe_table(x, y, smoothing=0.0).set_index("bin")
    assert t.loc["a", "woe"] == pytest.approx(np.log(0.8 / 0.2))
    assert t["iv_part"].sum() == pytest.approx((0.8 - 0.2) * np.log(4) * 2)
    assert A.iv_strength(0.01) == "not useful" and A.iv_strength(0.6) == "suspicious"


def test_cramers_v_identity_and_independence():
    a = pd.Series(list("xyz") * 200)
    assert A.cramers_v(a, a) == pytest.approx(1.0, abs=1e-3)
    rng = np.random.default_rng(0)
    assert A.cramers_v(pd.Series(rng.choice(list("ab"), 5000)), pd.Series(rng.choice(list("cd"), 5000))) < 0.05


def test_selection_drops_weak_and_redundant():
    iv = pd.DataFrame({"feature": ["m1", "m2", "c1", "c2", "weak"], "iv": [0.9, 0.8, 0.5, 0.4, 0.01]})
    sp = pd.DataFrame([[1, 0.95], [0.95, 1]], index=["m1", "m2"], columns=["m1", "m2"])
    cv = pd.DataFrame([[1, 0.9], [0.9, 1]], index=["c1", "c2"], columns=["c1", "c2"])
    kept, decisions = A.select_features(iv, sp, ["m1", "m2", "weak"], 0.02, 0.85, cv, 0.8)
    assert kept == ["m1", "c1"]
    assert set(decisions.loc[decisions.decision == "drop", "feature"]) == {"m2", "c2", "weak"}


def test_transformers_learn_from_training_only():
    train = pd.DataFrame({"n": list(range(100)), "c": ["a"] * 95 + ["b"] * 5})
    capper = QuantileCapper(["n"], 0.9).fit(train)
    assert capper.transform(pd.DataFrame({"n": [1000]}))["n"].iloc[0] == pytest.approx(train["n"].quantile(0.9))
    grouper = RareGrouper(["c"], 0.1).fit(train)
    assert grouper.transform(pd.DataFrame({"c": ["a", "b", "new"]}))["c"].tolist() == ["a", "Other (rare)",
                                                                                           "Other (rare)"]
    y = np.r_[np.zeros(50), np.ones(50)]
    enc = WoEEncoder(["n"], ["c"], bins=4).fit(train, y)
    out = enc.transform(pd.DataFrame({"n": [5, 95], "c": ["a", "zzz"]}))
    assert out.loc[0, "n"] < 0 < out.loc[1, "n"]
    assert out.loc[1, "c"] == 0.0  # unseen category carries no evidence
