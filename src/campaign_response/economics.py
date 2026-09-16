"""Bonus question: campaign profit on the 10,000-customer list.

Campaign assumptions:

* the confusion matrix on the list has the same percentages as on the
  test sample (the pipeline also reports the development sample's
  out-of-fold percentages);
* everyone called qualifies (100% approval);
* the risk-band mix (High / Medium / Low) applies equally to takers and
  non-takers, i.e. risk band is independent of the response prediction;
* a customer who is not called generates no profit and no cost.

Net profit per called customer is the Finance estimate for their risk band
if they take up the offer, and ``cost_not_taken_up`` if they do not.
"""

from __future__ import annotations

import pandas as pd

from .evaluation import value_per_sale


def scale_confusion(m: dict, size: int) -> dict:
    return {k: m[k] / m["n"] * size for k in ("tp", "fp", "tn", "fn")}


def campaign_table(m: dict, campaign: dict) -> pd.DataFrame:
    """Per risk band: expected customers in each confusion cell and the net profit."""
    size = campaign["list_size"]
    c = scale_confusion(m, size)
    cost = campaign["cost_not_taken_up"]
    rows = []
    for band, b in campaign["risk_bands"].items():
        s, win = b["share"], b["profit_taken_up"]
        rows.append({
            "risk_band": band, "share": s, "profit_taken_up": win, "profit_not_taken_up": cost,
            "called_takers_tp": c["tp"] * s, "called_non_takers_fp": c["fp"] * s,
            "missed_takers_fn": c["fn"] * s, "skipped_non_takers_tn": c["tn"] * s,
            "net_profit_called": c["tp"] * s * win + c["fp"] * s * cost,
            "missed_profit_fn": c["fn"] * s * win,
            "wasted_cost_fp": -c["fp"] * s * cost,
        })
    t = pd.DataFrame(rows)
    total = t.sum(numeric_only=True)
    total["share"], total["profit_taken_up"], total["profit_not_taken_up"] = 1.0, value_per_sale(campaign), cost
    t.loc[len(t)] = {"risk_band": "Total", **total.to_dict()}
    return t


def scenario_summary(m: dict, campaign: dict) -> dict:
    """Model-driven campaign vs calling everyone vs perfect targeting."""
    size = campaign["list_size"]
    c = scale_confusion(m, size)
    v, cost = value_per_sale(campaign), campaign["cost_not_taken_up"]
    takers, non_takers = c["tp"] + c["fn"], c["fp"] + c["tn"]
    model = c["tp"] * v + c["fp"] * cost
    perfect = takers * v
    everyone = takers * v + non_takers * cost
    return {
        "value_per_sale": v,
        "customers_called": c["tp"] + c["fp"],
        "expected_sales": c["tp"],
        "a_expected_net_profit": model,
        "b_lost_opportunity_total": perfect - model,
        "b_missed_sales_profit_fn": c["fn"] * v,
        "b_wasted_call_cost_fp": -c["fp"] * cost,
        "perfect_targeting_profit": perfect,
        "call_everyone_profit": everyone,
        "model_vs_call_everyone": model - everyone,
        "share_of_perfect_captured": model / perfect,
    }


def band_specific_cutoffs(y, p, campaign: dict) -> pd.DataFrame:
    """Optional refinement: one break-even cut-off per risk band.

    A take-up in the High band is worth less than a failed call costs at the
    single cut-off, so each band gets its own break-even probability
    cost / (profit + cost). Applied to the test score distribution, assuming
    (as in the campaign assumptions) that risk band is independent of the response score.
    """
    size, cost = campaign["list_size"], -campaign["cost_not_taken_up"]
    rows = []
    for band, b in campaign["risk_bands"].items():
        t = cost / (b["profit_taken_up"] + cost)
        call = p >= t
        tp, fp = (call & (y == 1)).sum(), (call & (y == 0)).sum()
        rows.append({
            "risk_band": band, "cutoff": t, "call_rate": call.mean(),
            "called": call.sum() / len(y) * size * b["share"],
            "net_profit": (tp * b["profit_taken_up"] - fp * cost) / len(y) * size * b["share"],
        })
    t = pd.DataFrame(rows)
    t.loc[len(t)] = {"risk_band": "Total", "cutoff": float("nan"), "call_rate": float("nan"),
                     "called": t["called"].sum(), "net_profit": t["net_profit"].sum()}
    return t
