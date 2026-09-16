import numpy as np
import pytest

from campaign_response import economics, evaluation as ev
from campaign_response.config import load_config
from campaign_response.pipeline import CANDIDATE_CATEGORICAL, CANDIDATE_NUMERIC


@pytest.fixture
def campaign():
    return load_config()["campaign"]


def test_value_per_sale_and_break_even(campaign):
    v = ev.value_per_sale(campaign)
    assert v == pytest.approx(0.10 * 285 + 0.25 * 705 + 0.65 * 1225)  # 1,001
    assert ev.break_even_probability(campaign) == pytest.approx(300 / (v + 300))


def test_campaign_numbers_by_hand(campaign):
    m = {"tp": 50, "fp": 150, "tn": 750, "fn": 50, "n": 1000}
    s = economics.scenario_summary(m, campaign)
    v = ev.value_per_sale(campaign)
    # scaled to 10,000: TP 500, FP 1,500, FN 500, TN 7,500
    assert s["customers_called"] == pytest.approx(2000)
    assert s["a_expected_net_profit"] == pytest.approx(500 * v - 1500 * 300)
    assert s["b_missed_sales_profit_fn"] == pytest.approx(500 * v)
    assert s["b_wasted_call_cost_fp"] == pytest.approx(1500 * 300)
    assert s["b_lost_opportunity_total"] == pytest.approx(500 * v + 1500 * 300)
    assert s["call_everyone_profit"] == pytest.approx(1000 * v - 9000 * 300)
    t = economics.campaign_table(m, campaign).set_index("risk_band")
    assert t.loc["Total", "net_profit_called"] == pytest.approx(s["a_expected_net_profit"])
    assert t.loc["High", "called_takers_tp"] == pytest.approx(50)


def test_profit_curve_prefers_break_even_region(campaign):
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)  # perfectly calibrated scores
    t = ev.best_threshold(ev.profit_curve(y, p, campaign))
    assert abs(t - ev.break_even_probability(campaign)) <= 0.03


def test_metrics_helpers():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    assert ev.ks_statistic(y, p) == pytest.approx(0.5)
    assert ev.lift_at(y, p, 0.25) == pytest.approx(2.0)
    assert ev.psi(p, p) == pytest.approx(0.0, abs=1e-9)


def test_post_call_field_is_never_a_candidate():
    assert "call_duration_s" not in CANDIDATE_NUMERIC + CANDIDATE_CATEGORICAL


def test_band_cutoffs_are_break_even_per_band(campaign):
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.6, 0.25, 0.1])
    t = economics.band_specific_cutoffs(y, p, campaign).set_index("risk_band")
    assert t.loc["High", "cutoff"] == pytest.approx(300 / 585)
    assert t.loc["Low", "cutoff"] == pytest.approx(300 / 1525)
    # High band calls only p >= 0.513: one taker (0.9) and one non-taker (0.6)
    assert t.loc["High", "net_profit"] == pytest.approx((285 - 300) / 4 * 10000 * 0.10)
