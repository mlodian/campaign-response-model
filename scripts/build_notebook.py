"""Generate notebooks/campaign_response_walkthrough.ipynb.

The notebook follows the project's design questions, in order, and
answers each one from the pipeline's saved outputs.
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# Campaign response model: walkthrough

**Goal.** Rank customers by how likely they are to take up the offer when cold-called, so call effort goes where it
converts. **Secondary goal.** Translate the model's classification into expected campaign profit.

Each section below answers one design question. Code lives in `src/campaign_response/`; this notebook
reads the outputs of `make pipeline` (set `RERUN = True` to rebuild them, roughly 15 minutes)."""),
    code("""import json

import joblib
import numpy as np
import pandas as pd
from IPython.display import Image, display

from campaign_response.config import load_config
from campaign_response.data import load_raw
from campaign_response.pipeline import step_evaluate, step_figures, step_periods, step_prepare, step_train

RERUN = False
cfg = load_config()
R, F = cfg.reports, cfg.figures
pd.set_option("display.max_columns", 30, "display.width", 160, "display.float_format", "{:,.4f}".format)
show = lambda name, width=820: display(Image(filename=F / name, width=width))
if RERUN:
    for step in (step_prepare, step_train, step_figures, step_evaluate, step_periods):
        step(cfg)"""),
    md("## 0. The data as delivered"),
    code("""raw = load_raw(cfg.raw_file, cfg["data"]["separator"])
print(raw.shape)
raw.dtypes.value_counts()  # client records are not printed; see data/README.md for the dictionary"""),
    code("""pd.read_csv(R / "data_profile.csv")"""),
    md("""## 1. Are all input variables relevant?

No. Two kinds of variables are removed:
- **Leaks.** Call duration is only known after the call, according to the data dictionary.
- **Variables with no discriminating power**, measured by Information Value (IV) on the training split. The
  decision log below records every keep or drop."""),
    code("""pd.read_csv(R / "feature_decisions.csv")"""),
    md("""## 2. What does the data look like? Outliers and corrections

The table below lists each audit step and how many rows or cells it touched. Call attempts has a long tail
(capped at the training 99th percentile inside the model); age extremes (up to 98) are plausible and kept.
`days_since_prev_campaign = 999` is a code for "never contacted", not a number of days, so it is replaced by a
recency band."""),
    code("""pd.read_csv(R / "audit_log.csv")"""),
    code("""show("02_outliers.png")
show("01_leakage_call_duration.png")"""),
    md("""## 3. Is there discriminating power among the variables?

Each candidate variable is scored on the training split with four measures:
- Weight of Evidence (WoE) and Information Value (IV). Rule-of-thumb bands (derived from Siddiqi): below 0.02 not useful, 0.02–0.1
  weak, 0.1–0.3 medium, 0.3–0.5 strong, above 0.5 suspicious.
- Univariate Gini.
- Kolmogorov-Smirnov (KS), for numeric variables.
- Chi-square with Cramér's V, for categorical variables.

The macro indicators score as "suspicious" because they mark campaign periods with very different take-up.
They are known before the call, so they are kept, and their drift risk is noted."""),
    code("""pd.read_csv(R / "information_value.csv")"""),
    code("""show("03_information_value.png")
show("07_rate_by_month.png")
show("08_rate_by_previous_outcome.png")
show("09_macro_number_of_employees.png")"""),
    md("""## 4. How are high correlations controlled?

- **Numeric variables.** Spearman correlation and variance inflation factors (VIF). Within any group with
  |ρ| > 0.85, only the variable with the highest IV is kept.
- **Categorical variables.** Bias-corrected Cramér's V; pairs above 0.8 are treated as redundant.
- **After selection.** Every VIF must be below 10, or the pipeline fails."""),
    code("""display(pd.read_csv(R / "vif_before.csv"))
display(pd.read_csv(R / "vif_after.csv"))"""),
    code("""show("04_spearman_numeric.png", 640)
show("05_cramers_v_categorical.png", 700)"""),
    md("""## 5. Should the response classes be balanced?

Take-up is 11.2%. The data is **not** resampled, for three reasons:
- Resampling distorts predicted probabilities.
- The profit cut-off relies on calibrated probabilities.
- The evaluation metrics already handle imbalance: PR-AUC, lift, KS and profit.

Class weights were not needed; the cut-off does the balancing."""),
    code("""json.loads((R / "data_summary.json").read_text())"""),
    md("""## 6. Different scales, encoding, grouping and binning

| Question | Logistic models | Tree models |
|---|---|---|
| Different scales | Standardised (one-hot model), or WoE puts every input on the log-odds scale | Not needed |
| Categorical encoding | WoE (scorecard) or one-hot | One-hot |
| Grouping | Levels below 1% of training rows are merged into "Other (rare)" | Same |
| Binning numerics | Yes for the scorecard: decile bins capture non-linear effects such as age | No; trees find their own splits |

All encoders are fitted on training folds only."""),
    code("""show("06_woe_age.png")"""),
    md("""## 7. Train/test split and validation checks while training

- **Split.** 70/30, stratified on the response. The test set is scored once, at the end.
- **Validation.** Every model is validated with 5-fold stratified CV on the training data. Preprocessing is
  re-fitted inside each fold.
- **Tuning.** Uses its own 3-fold CV.
- **Cut-off.** Chosen on out-of-fold training scores.
- **Campaign periods.** A grouped CV holds out whole periods (section 15)."""),
    md("""## 8. Choosing the best model (at least one non-linear)

The model ladder has six rungs:
1. baseline;
2. business rule;
3. WoE logistic scorecard;
4. one-hot logistic regression;
5. **random forest** (non-linear);
6. **LightGBM** (non-linear).

Models are ranked on CV PR-AUC (the primary metric), with Gini, KS, top-decile lift and Brier score reported
alongside."""),
    code("""cv = pd.read_csv(R / "cv_summary.csv")
cv[["model", "pr_auc_mean", "pr_auc_std", "gini_mean", "ks_mean", "lift_top10_mean", "brier_mean",
    "overfit_gap_pr_auc"]]"""),
    code("""show("10_model_ladder_cv.png", 900)"""),
    md("""## 9. Which models to optimise, and how

The two best untuned models on CV PR-AUC go forward and are both tuned with a randomised search (25 candidates,
3-fold CV, scored on PR-AUC):
- **Randomised search** covers a large, mixed parameter space efficiently.
- **PR-AUC** is the target because the business needs a clean top of the call list.

The tuned finalist with the higher 5-fold CV PR-AUC is selected. The table after the search results compares
models fold by fold: the non-linear models effectively tie, and tuning reuses the training folds (no nested CV),
so the small tuning gains should not be over-read."""),
    code("""json.loads((R / "best_params.json").read_text())"""),
    code("""pd.read_csv(R / "cv_paired_differences.csv")"""),
    code("""pd.read_csv(R / "tuning_results.csv")[["model", "rank_test_score", "mean_test_score", "std_test_score"]] \\
    .sort_values(["model", "rank_test_score"]).groupby("model").head(3)"""),
    md("""## 10. Guarding against over- and underfitting

Four checks:
- the gap between training-fold and validation-fold scores;
- a learning curve;
- the LightGBM boosting curve (validation loss by round);
- score stability between the out-of-fold and test samples (Population Stability Index, PSI)."""),
    code("""for f in ["11_train_vs_validation.png", "12_learning_curve.png", "13_boosting_curve.png", "19_score_stability.png"]:
    if (F / f).exists():
        show(f, 760)"""),
    md("""## 11. Interpreting the predictors, and most and least important inputs

- **SHAP** shows how the selected model uses each input.
- **Permutation importance** shows how much the test PR-AUC drops when an input is shuffled.
- **Scorecard coefficients** give the WoE model's view.

The least important inputs are listed at the bottom of each table."""),
    code("""display(pd.read_csv(R / "shap_importance.csv"))
display(pd.read_csv(R / "permutation_importance.csv"))
display(pd.read_csv(R / "scorecard_coefficients.csv"))"""),
    code("""show("20_feature_importance.png", 900)
show("21_shap_beeswarm.png", 760)"""),
    md("""## 12. Choosing the cut-off

Finance's figures give the economics:
- A take-up is worth **1,001** on average, weighting High, Medium and Low risk by 10%, 25% and 65%.
- A call that does not convert costs **300**.

Calling is worthwhile when p × 1,001 > (1 − p) × 300, i.e. p > 0.231 (Elkan, 2001). The cut-off that
maximises profit on out-of-fold scores is used, and compared with 0.5, the F1-optimal and the Youden-J cut-offs.
The profit curve is flat near its peak (388.3k at 0.20, 386.4k at 0.23), so the exact choice matters little."""),
    code("""json.loads((R / "thresholds.json").read_text())"""),
    code("""show("14_profit_vs_cutoff.png")"""),
    md("## 13. Final model on the held-out test set"),
    code("""m = json.loads((R / "test_metrics.json").read_text())
pd.Series({k: v for k, v in m.items() if not isinstance(v, dict)})"""),
    code("""pd.DataFrame(m["alternative_cutoffs_test"]).T"""),
    code("""for f in ["15_gains_lift.png", "16_ks.png", "17_calibration.png"]:
    show(f, 820)
pd.read_csv(R / "gains_table.csv")"""),
    md("""## 14. Reading the confusion matrix

- **True positives:** called, and took up the offer; this is revenue.
- **False positives:** called, but did not take up; each costs 300.
- **False negatives:** not called, but would have taken up; this is missed profit.
- **True negatives:** not called, and would not have taken up; the call cost is saved."""),
    code("""show("18_confusion_matrix.png", 600)"""),
    md("""## 15. Campaign periods: when versus whom

The four macro indicators take one value combination per campaign period (26 periods, one contact month each), so
a random split tests on periods the model has already seen. Two further checks:
- grouped CV that holds out whole periods;
- ranking *within* a period, since a list is called within one period.

Most of the lift comes from telling good periods from bad ones. Ranking within a period is modest."""),
    code("""display(pd.read_csv(R / "period_validation.csv"))
json.loads((R / "period_summary.json").read_text())"""),
    code("""show("22_period_breakdown.png", 900)
pd.read_csv(R / "period_breakdown.csv")"""),
    md("""## 16. Bonus: campaign economics on the 10,000-customer list

Assumptions:
- The list has the same confusion-matrix percentages as the test sample, the one the model never saw. The same
  answers from the development sample's out-of-fold rates are shown next to it (388k and 736k). As section 15
  shows, this assumption holds only for a list drawn from the same mix of periods.
- The per-sale profits are already net of the call cost; the 300 applies only when there is no take-up.
- Every customer called qualifies.
- The risk-band mix (10/25/65%) applies equally to predicted takers and non-takers.
- Customers who are not called cost nothing and earn nothing.

Questions:
- **(a)** Expected net profit = called takers × band profit + called non-takers × (−300).
- **(b)** Lost opportunity from misclassification = profit a perfect model would earn − this model's profit.
  This equals missed sales (false negatives × band profit) plus wasted calls (false positives × 300). Read
  narrowly as missed sales only, it is the first component (about 472k)."""),
    code("""econ = pd.read_csv(R / "campaign_economics.csv")
econ"""),
    code("""scen = json.loads((R / "campaign_scenarios.json").read_text())
dev = scen.pop("development_sample")
pd.DataFrame({"test sample": pd.Series(scen), "development sample (out-of-fold)": pd.Series(dev)})"""),
    code("""pd.read_csv(R / "campaign_economics_development.csv")"""),
    md("## 17. Scoring a new list"),
    code("""bundle = joblib.load(cfg.models / "response_model.joblib")
sample = raw.sample(5, random_state=3)
from campaign_response.cleaning import clean
prepared, _ = clean(sample, cfg["leakage_columns"])
prepared = prepared.assign(take_up_probability=bundle["pipeline"].predict_proba(prepared[bundle["features"]])[:, 1])
prepared["call"] = prepared["take_up_probability"] >= bundle["threshold"]
prepared[["take_up_probability", "call", "response"]].reset_index(drop=True)"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
})
out = ROOT / "notebooks" / "campaign_response_walkthrough.ipynb"
nbf.write(nb, out)
print("wrote", out)
