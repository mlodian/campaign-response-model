"""End-to-end pipeline: prepare -> train -> figures -> evaluate -> periods.

Usage:
    python -m campaign_response.pipeline --steps all
    python -m campaign_response.pipeline --steps prepare
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from . import analysis as A
from . import eda, economics, evaluation as ev, explain, periods as per
from .cleaning import TARGET, clean, profile
from .config import Config, load_config
from .data import load_raw
from .modeling import (
    boosting_curve, cross_validate_ladder, learning_curve_table, leakage_benchmark, model_ladder, out_of_fold,
    paired_differences, summarise_cv, tune,
)

log = logging.getLogger("campaign_response")
STEPS = ["prepare", "train", "figures", "evaluate", "periods"]

CANDIDATE_NUMERIC = ["age", "call_attempts", "prev_campaign_attempts", "employment_variation_rate",
                     "consumer_price_index", "consumer_confidence_index", "interbank_rate_3m", "number_of_employees"]
CANDIDATE_CATEGORICAL = ["employment_type", "civil_status", "education", "credit_in_default", "has_home_loan",
                         "has_personal_loan", "contact_medium", "last_contact_month", "last_contact_weekday",
                         "previous_outcome", "prev_contact_recency", "previously_contacted"]
MODEL_SELECTION_EXCLUDES = ("Baseline", "Business")


def split(df: pd.DataFrame, cfg: Config):
    s = cfg["split"]
    return train_test_split(df, test_size=s["test_size"], stratify=df[TARGET], random_state=s["random_state"])


def selected_features(cfg: Config) -> list[str]:
    return json.loads((cfg.reports / "selected_features.json").read_text())["features"]


def step_prepare(cfg: Config) -> None:
    raw = load_raw(cfg.raw_file, cfg["data"]["separator"])
    profile(raw).to_csv(cfg.reports / "data_profile.csv", index=False)
    df, audit = clean(raw, cfg["leakage_columns"])
    audit.to_frame().to_csv(cfg.reports / "audit_log.csv", index=False)
    df.to_parquet(cfg.clean_file, index=False)

    train, _ = split(df, cfg)
    fs = cfg["feature_selection"]
    candidates = CANDIDATE_NUMERIC + CANDIDATE_CATEGORICAL
    iv = A.information_values(train, candidates + ["leak_call_duration_s"], TARGET)
    iv.to_csv(cfg.reports / "information_value.csv", index=False)
    spearman = train[CANDIDATE_NUMERIC].corr(method="spearman")
    spearman.to_csv(cfg.reports / "spearman_numeric.csv")
    cramers = A.cramers_matrix(train, CANDIDATE_CATEGORICAL)
    cramers.to_csv(cfg.reports / "cramers_v_categorical.csv")
    A.vif_table(train, CANDIDATE_NUMERIC).to_csv(cfg.reports / "vif_before.csv", index=False)

    kept, decisions = A.select_features(iv[iv.feature.isin(candidates)], spearman, CANDIDATE_NUMERIC,
                                        fs["min_information_value"], fs["max_abs_spearman"],
                                        cramers, fs["max_cramers_v"])
    decisions.to_csv(cfg.reports / "feature_decisions.csv", index=False)
    kept_num = [f for f in kept if f in CANDIDATE_NUMERIC]
    vif_after = A.vif_table(train, kept_num)
    vif_after.to_csv(cfg.reports / "vif_after.csv", index=False)
    if (vif_after["vif"] > fs["max_vif"]).any():
        raise ValueError(f"VIF above {fs['max_vif']} after selection:\n{vif_after}")
    (cfg.reports / "selected_features.json").write_text(json.dumps(
        {"features": kept, "numeric": kept_num, "categorical": [f for f in kept if f not in kept_num]}, indent=2))
    log.info("Selected %d of %d candidate features; max VIF after %.1f", len(kept), len(candidates),
             vif_after["vif"].max())

    f = cfg.figures
    eda.plot_leakage(df, f / "01_leakage_call_duration.png")
    eda.plot_outliers(df, f / "02_outliers.png", cfg["outliers"]["cap_quantile"])
    eda.plot_iv(iv, f / "03_information_value.png")
    eda.plot_heatmap(spearman, f / "04_spearman_numeric.png",
                     "Three macro indicators move together (|rho| > 0.9)", "Spearman correlation, training split",
                     diverging=True)
    eda.plot_heatmap(cramers, f / "05_cramers_v_categorical.png",
                     "Previous-contact fields repeat the previous-outcome signal",
                     "Bias-corrected Cramer's V, training split", diverging=False)
    eda.plot_woe_curve(train, "age", f / "06_woe_age.png",
                       "Age has a U-shaped effect, so the scorecard bins it")
    eda.plot_rate(df, "last_contact_month", f / "07_rate_by_month.png",
                  "Take-up swings from 6% to over 40% depending on the contact month",
                  "Take-up by last contact month (masked labels)",
                  highlight={"Cat_8_o***t", "Cat_9_s***p", "Cat_5_m***r", "Cat_2_d***c"})
    eda.plot_rate(df, "previous_outcome", f / "08_rate_by_previous_outcome.png",
                  "Customers who took up a previous offer convert at about 6x the average",
                  "Take-up by previous campaign outcome", highlight={"Cat_2_s***s"}, height=3.2)
    eda.plot_macro(df, f / "09_macro_number_of_employees.png")
    summary = {"rows": len(df), "take_up_rate": float(df[TARGET].mean()), "features_selected": kept}
    (cfg.reports / "data_summary.json").write_text(json.dumps(summary, indent=2))


def step_train(cfg: Config) -> None:
    df = pd.read_parquet(cfg.clean_file)
    train, _ = split(df, cfg)
    feats = selected_features(cfg)
    X, y = train[feats], train[TARGET].to_numpy()
    seed, folds = cfg["split"]["random_state"], cfg["cv"]["folds"]
    fs, cap_q = cfg["feature_selection"], cfg["outliers"]["cap_quantile"]

    ladder = model_ladder(feats, CANDIDATE_NUMERIC, fs, cap_q)
    cv_folds = cross_validate_ladder(ladder, X, y, folds, seed)
    summary = summarise_cv(cv_folds)
    contenders = summary[~summary["model"].str.startswith(MODEL_SELECTION_EXCLUDES)]
    finalists = contenders.sort_values("pr_auc_mean", ascending=False)["model"].head(2).tolist()
    log.info("Finalists for tuning: %s", finalists)

    tuned, tuning_rows = {}, []
    for name in finalists:
        search = tune(ladder[name], name, X, y, cfg["tuning"]["n_iter"], cfg["tuning"]["folds"], seed)
        res = pd.DataFrame(search.cv_results_).assign(model=name)
        tuning_rows.append(res)
        tuned[f"{name} (tuned)"] = (clone(ladder[name]).set_params(**search.best_params_), search.best_params_)
    pd.concat(tuning_rows).sort_values(["model", "rank_test_score"]).to_csv(
        cfg.reports / "tuning_results.csv", index=False)
    tuned_folds = cross_validate_ladder({k: v[0] for k, v in tuned.items()}, X, y, folds, seed)
    cv_folds = pd.concat([cv_folds, tuned_folds], ignore_index=True)
    cv_folds.to_csv(cfg.reports / "cv_folds.csv", index=False)
    summary = summarise_cv(cv_folds)
    summary.to_csv(cfg.reports / "cv_summary.csv", index=False)
    log.info("\n%s", summary[["model", "pr_auc_mean", "roc_auc_mean", "ks_mean", "lift_top10_mean",
                              "overfit_gap_pr_auc"]].round(4).to_string(index=False))
    tuned_summary = summary[summary["model"].isin(tuned)]
    tuned_name = tuned_summary.loc[tuned_summary["pr_auc_mean"].idxmax(), "model"]
    final, best_params = tuned[tuned_name]
    log.info("Selected model: %s", tuned_name)

    oof = out_of_fold(final, X, y, folds, seed)
    curve = ev.profit_curve(y, oof, cfg["campaign"])
    curve.to_csv(cfg.reports / "profit_curve_oof.csv", index=False)
    threshold = ev.best_threshold(curve)
    alternatives = ev.alternative_thresholds(y, oof)
    alternatives["break_even"] = ev.break_even_probability(cfg["campaign"])
    alternatives["profit_optimal_oof"] = threshold
    (cfg.reports / "thresholds.json").write_text(json.dumps(alternatives, indent=2))
    np.save(cfg.models / "oof_scores.npy", oof)

    learning_curve_table(final, X, y, folds, seed).to_csv(cfg.reports / "learning_curve.csv", index=False)
    gbm = next((pipe for name, (pipe, _) in tuned.items() if "LightGBM" in name), None)
    if gbm is not None:
        boosting_curve(gbm, X, y, seed).to_csv(cfg.reports / "boosting_curve.csv", index=False)
    leakage_benchmark(feats, CANDIDATE_NUMERIC, fs, cap_q, train, y, folds, seed).to_csv(
        cfg.reports / "leakage_benchmark.csv", index=False)

    final.fit(X, y)
    scorecard = ladder["Logistic scorecard (WoE)"].fit(X, y)
    joblib.dump({"pipeline": final, "name": tuned_name, "threshold": threshold, "features": feats,
                 "params": best_params}, cfg.models / "response_model.joblib")
    joblib.dump(scorecard, cfg.models / "woe_scorecard.joblib")
    (cfg.reports / "best_params.json").write_text(json.dumps(
        {"model": tuned_name, **best_params}, indent=2, default=float))
    log.info("Saved %s, profit-optimal cut-off %.2f", tuned_name, threshold)


def step_figures(cfg: Config) -> None:
    """Redraw training-stage charts from saved results."""
    f = cfg.figures
    summary = pd.read_csv(cfg.reports / "cv_summary.csv")
    selected = json.loads((cfg.reports / "best_params.json").read_text())["model"]
    paired_differences(pd.read_csv(cfg.reports / "cv_folds.csv"), selected).to_csv(
        cfg.reports / "cv_paired_differences.csv", index=False)
    ev.plot_ladder(summary, f / "10_model_ladder_cv.png")
    ev.plot_overfit(summary, f / "11_train_vs_validation.png")
    ev.plot_learning_curve(pd.read_csv(cfg.reports / "learning_curve.csv"), f / "12_learning_curve.png")
    if (cfg.reports / "boosting_curve.csv").exists():
        ev.plot_boosting_curve(pd.read_csv(cfg.reports / "boosting_curve.csv"), f / "13_boosting_curve.png")
    thr = json.loads((cfg.reports / "thresholds.json").read_text())
    ev.plot_profit(pd.read_csv(cfg.reports / "profit_curve_oof.csv"), thr["profit_optimal_oof"], thr["break_even"],
                   thr, f / "14_profit_vs_cutoff.png")


def step_evaluate(cfg: Config) -> None:
    df = pd.read_parquet(cfg.clean_file)
    train, test = split(df, cfg)
    bundle = joblib.load(cfg.models / "response_model.joblib")
    model, threshold, feats = bundle["pipeline"], bundle["threshold"], bundle["features"]
    y = test[TARGET].to_numpy()
    p = model.predict_proba(test[feats])[:, 1]
    oof = np.load(cfg.models / "oof_scores.npy")

    m = ev.classification_metrics(y, p, threshold)
    m["model"] = bundle["name"]
    m["psi_train_oof_vs_test"] = ev.psi(oof, p)
    m["value_per_sale"] = ev.value_per_sale(cfg["campaign"])
    scorecard = joblib.load(cfg.models / "woe_scorecard.joblib")
    ps = scorecard.predict_proba(test[feats])[:, 1]
    m["scorecard_test"] = {k: v for k, v in ev.classification_metrics(y, ps, threshold).items()
                           if k in ("pr_auc", "roc_auc", "gini", "ks", "lift_top10", "brier")}
    alt = json.loads((cfg.reports / "thresholds.json").read_text())
    m["alternative_cutoffs_test"] = {
        name: {k: v for k, v in ev.classification_metrics(y, p, t).items()
               if k in ("threshold", "precision", "recall", "call_rate", "tp", "fp", "fn", "tn")}
        for name, t in alt.items() if name != "base_rate"
    }
    (cfg.reports / "test_metrics.json").write_text(json.dumps(m, indent=2, default=float))
    log.info("Test: PR-AUC %.3f ROC-AUC %.3f KS %.3f lift@10 %.2f precision %.3f recall %.3f",
             m["pr_auc"], m["roc_auc"], m["ks"], m["lift_top10"], m["precision"], m["recall"])

    gains = ev.gains_table(y, p)
    gains.to_csv(cfg.reports / "gains_table.csv", index=False)
    econ = economics.campaign_table(m, cfg["campaign"])
    econ.to_csv(cfg.reports / "campaign_economics.csv", index=False)
    scen = economics.scenario_summary(m, cfg["campaign"])
    (cfg.reports / "campaign_scenarios.json").write_text(json.dumps(scen, indent=2, default=float))
    bands = economics.band_specific_cutoffs(y, p, cfg["campaign"])
    bands.to_csv(cfg.reports / "campaign_band_cutoffs.csv", index=False)
    scen["band_cutoffs_net_profit"] = float(bands["net_profit"].iloc[-1])
    scen["band_cutoffs_customers_called"] = float(bands["called"].iloc[-1])
    # The same answers from the development sample's out-of-fold confusion matrix
    m_dev = ev.classification_metrics(train[TARGET].to_numpy(), oof, threshold)
    economics.campaign_table(m_dev, cfg["campaign"]).to_csv(
        cfg.reports / "campaign_economics_development.csv", index=False)
    scen["development_sample"] = economics.scenario_summary(m_dev, cfg["campaign"])
    (cfg.reports / "campaign_scenarios.json").write_text(json.dumps(scen, indent=2, default=float))
    log.info("Campaign: expected net profit %.0f; lost opportunity %.0f; band cut-offs %.0f",
             scen["a_expected_net_profit"], scen["b_lost_opportunity_total"], scen["band_cutoffs_net_profit"])
    pd.DataFrame({"response": y, "score": p, "call": (p >= threshold).astype(int)}).to_csv(
        cfg.reports / "test_predictions.csv.gz", index=False)

    f = cfg.figures
    ev.plot_gains(gains, f / "15_gains_lift.png")
    ev.plot_ks(y, p, f / "16_ks.png")
    ev.plot_calibration(y, p, f / "17_calibration.png")
    ev.plot_confusion(m, f / "18_confusion_matrix.png")
    ev.plot_psi(oof, p, m["psi_train_oof_vs_test"], f / "19_score_stability.png")

    seed = cfg["split"]["random_state"]
    values, Xt = explain.shap_values(model, test[feats], cfg["explain"]["shap_sample"], seed, feats)
    shap_t = explain.shap_by_feature(values, Xt, feats)
    shap_t.to_csv(cfg.reports / "shap_importance.csv", index=False)
    perm = explain.permutation_table(model, test[feats], y, seed)
    perm.to_csv(cfg.reports / "permutation_importance.csv", index=False)
    explain.plot_importance(shap_t, perm, f / "20_feature_importance.png")
    explain.plot_beeswarm(values, Xt, f / "21_shap_beeswarm.png")
    explain.scorecard_table(scorecard).to_csv(cfg.reports / "scorecard_coefficients.csv", index=False)


def step_periods(cfg: Config) -> None:
    """Hold out whole campaign periods, and measure ranking within a period."""
    df = pd.read_parquet(cfg.clean_file)
    df["period"] = per.period_ids(df)
    train, test = split(df, cfg)
    bundle = joblib.load(cfg.models / "response_model.joblib")
    feats, params = bundle["features"], bundle["params"]
    customer_feats = [f for f in feats if f not in per.TIMING_FEATURES]
    fs, cap_q = cfg["feature_selection"], cfg["outliers"]["cap_quantile"]
    customer_model = model_ladder(customer_feats, CANDIDATE_NUMERIC, fs, cap_q)["LightGBM"].set_params(**params)
    variants = {bundle["name"]: (bundle["pipeline"], feats),
                "Customer-only LightGBM (no timing inputs)": (customer_model, customer_feats)}
    seed, folds = cfg["split"]["random_state"], cfg["cv"]["folds"]
    table = per.validation_table(variants, train, test, TARGET, folds, seed)
    table.to_csv(cfg.reports / "period_validation.csv", index=False)
    log.info("\n%s", table.round(3).to_string(index=False))

    y = test[TARGET].to_numpy()
    p = bundle["pipeline"].predict_proba(test[feats])[:, 1]
    breakdown = per.period_breakdown(test, y, p, bundle["threshold"], cfg["campaign"])
    breakdown.to_csv(cfg.reports / "period_breakdown.csv", index=False)
    summary = {"periods": int(df["period"].nunique()),
               "months_per_period_max": int(df.groupby("period")["last_contact_month"].nunique().max()),
               "periods_with_take_up_above_break_even": int(
                   (breakdown["take_up_rate"] >= ev.break_even_probability(cfg["campaign"])).sum())}
    (cfg.reports / "period_summary.json").write_text(json.dumps(summary, indent=2))
    per.plot_breakdown(breakdown, ev.break_even_probability(cfg["campaign"]), cfg.figures / "22_period_breakdown.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="+", default=["all"], choices=STEPS + ["all"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config()
    for step in (STEPS if "all" in args.steps else [s for s in STEPS if s in args.steps]):
        t = time.time()
        log.info("== %s", step)
        globals()[f"step_{step}"](cfg)
        log.info("== %s done in %.0fs", step, time.time() - t)


if __name__ == "__main__":
    main()
