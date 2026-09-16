# Method rationale: review of related literature

This review ties each modelling choice to published practice. Numbers marked **(this project)** come from `reports/` (70/30 stratified split of 34,989 de-duplicated rows, 11.2% take-up). Machine-readable sources: [`references.csv`](references.csv). Every source and every attributed claim was checked independently against the publisher, the DOI registry or the source document.

---

## 1. The problem and the data

**Telemarketing response modelling.** Predicting whether a contacted bank customer subscribes to an offer is an established decision-support problem. Moro, Cortez & Rita (2014) predicted the success of a Portuguese bank's telemarketing calls (data from 2008–2013). They compared logistic regression, decision trees, neural networks and support vector machines, and reached 79% of subscribers by calling the top-ranked half of clients.

**Relation to a public dataset.** The client file's structure resembles the public UCI "Bank Marketing" dataset:
- 20 inputs, grouped as bank-client, last-contact, other campaign attributes and social/economic context;
- the same kinds of variables (employment, marital status, education, default, housing and personal loans, contact medium, month, weekday, previous outcome and five macro indicators);
- a 999 code for "not previously contacted";
- an ~11% positive rate (11.3% in the public file) [Moro, Rita & Cortez, 2014, UCI dataset]. The public subset covers May 2008 to November 2010.

The labels here are masked and the row count differs (35,000), so this is noted only as context. The analysis uses the client file as supplied.

**Call duration is a leak.** The client's data dictionary states that call duration is "only known after call has taken place". The UCI documentation for the public dataset makes the same point: duration strongly affects the outcome, should be used only for benchmarking, and should be discarded for a realistic predictive model. Using information that would not be legitimately available at prediction time is a textbook case of leakage [Kaufman et al., 2012].

**This project.**
- Duration has the highest Information Value (1.93).
- A cross-validated LightGBM reaches PR-AUC 0.641 with duration and 0.445 without it.
- The deployable model excludes duration.

## 2. Variable relevance: Weight of Evidence and Information Value

Statistical classification is the basis of consumer credit scoring [Hand & Henley, 1997], and WoE/IV binning is the standard way scorecard developers screen characteristics and encode them for logistic regression [Siddiqi, 2006]. A commonly quoted rule of thumb, derived from Siddiqi, is:

| IV | Predictive power |
|---|---|
| below 0.02 | unpredictive |
| 0.02–0.1 | weak |
| 0.1–0.3 | medium |
| 0.3–0.5 | strong |
| above 0.5 | suspicious |

Siddiqi gives the first four bands (above 0.3 = strong) and advises checking values above 0.5 for over-prediction. The "suspicious" label comes from later practitioner sources.

**This project.**
- Weekday (IV 0.008) and the home-loan and personal-loan flags (0.001 each) fall below 0.02 and are dropped.
- Four of the five macro indicators score above 0.5 (the consumer price index scores 0.47). They are known before the call, so they are kept but flagged for drift monitoring.
- IV is complemented by univariate Gini, KS and chi-square tests.

## 3. Redundancy: correlation, VIF and Cramér's V

**Numeric variables.** Variance inflation factors diagnose multicollinearity in linear models. The rule of thumb "VIF above 10 is a problem" is widely used, but O'Brien (2007) warns that even VIFs of 10, 20 or 40 do not by themselves justify dropping variables. Here, variables are dropped because they are near-duplicates (Spearman ≈ 0.94) with lower IV. VIF only confirms the result.

**Categorical variables.** Association is measured with Cramér's V, using Bergsma's (2013) bias correction, because the uncorrected statistic is biased upward.

**This project.**
- Three macro indicators have Spearman correlations of 0.93–0.94 and VIFs of 32–64. The one with the highest IV (number of employees) is kept, which brings the maximum VIF down to 1.8.
- "Previously contacted" duplicates the recency band (Cramér's V 1.00) and is dropped.

**Campaign periods.** When observations come in groups that share conditions, random cross-validation mixes the groups between folds and overstates performance on new groups. Blocking the folds by group gives a more honest estimate [Roberts et al., 2017]. The UCI documentation lists the macro indicators as monthly or quarterly values (the interbank rate is daily), so rows with the same values belong to the same period.

**This project.**
- The four macro indicators identify 26 campaign periods, each with a single contact month.
- Holding out whole periods lowers pooled ROC-AUC from 0.81 to 0.72.
- Within a single period, ROC-AUC is about 0.55, so the macro inputs mainly signal *when* a campaign pays.

## 4. Class imbalance

In simulations with logistic regression, random under- or over-sampling and SMOTE produced strongly miscalibrated risks without improving AUC [van den Goorbergh et al., 2022]. With 11% take-up, this project keeps the natural class ratio. It evaluates with precision-recall analysis, summarised by PR-AUC, which is more informative than ROC analysis on imbalanced data [Saito & Rehmsmeier, 2015], and it sets the decision cut-off from costs.

**This project.** The test Brier score is 0.077, and calibration by decile tracks the diagonal.

## 5. Model comparison

- **Logistic regression versus ensembles.** A large credit-scoring benchmark finds that several classifiers beat the industry-standard logistic regression, with heterogeneous ensembles performing best. Logistic regression remains the industry-standard benchmark [Lessmann et al., 2015].
- **The non-linear candidates.**
  - Random forests average decorrelated trees [Breiman, 2001].
  - LightGBM is a gradient-boosting decision tree implementation built for efficiency on large tabular data [Ke et al., 2017].

**This project.**
- **Cross-validated PR-AUC:** WoE scorecard 0.423, one-hot logistic 0.439, random forest 0.453, LightGBM 0.445.
- **After tuning:** LightGBM 0.458, random forest 0.454; LightGBM is selected.

## 6. Hyper-parameter optimisation

Random search is more efficient than grid search when only a few hyper-parameters matter [Bergstra & Bengio, 2012]. The two best untuned models are tuned with 25 random candidates each, scored by 3-fold cross-validated PR-AUC.

**This project.** Tuning cut LightGBM's train-validation PR-AUC gap from 0.16 to 0.04. It did this with fewer leaves (13) and a lower learning rate (0.01), which also addresses overfitting.

## 7. Probabilities, calibration and the cut-off

- **Calibration.** Boosted trees tend to produce distorted (sigmoid-shaped) probabilities, and the picture for random forests is less clear-cut, so calibration should be checked [Niculescu-Mizil & Caruana, 2005].
- **Cut-off from costs.** With calibrated probabilities, the cost-optimal decision is to act when p exceeds c_FP / (c_FP + c_FN) [Elkan, 2001].
- **Profit-driven evaluation.** In customer analytics, evaluating models by the profit they generate, rather than by accuracy, is recommended [Verbeke et al., 2012].

**This project.**
- Finance's figures give a take-up value of 1,001 (the average of 285, 705 and 1,225, weighted by the band mix) and a failed-call cost of 300, so the break-even probability is 300 / 1,301 = 0.23.
- The cut-off that maximises profit on out-of-fold training scores is 0.20.
- Per-band break-even cut-offs (High 0.51, Medium 0.30, Low 0.20) are shown as a refinement.

## 8. Stability and interpretation

**Stability.** The Population Stability Index (PSI) compares score distributions between samples. By a widely used rule of thumb, PSI below 0.1 indicates little change. Yurdakul & Naranjo (2020) note that this cut-off ignores sample size and propose sample-size-dependent benchmarks.

**This project.** PSI between out-of-fold training scores and test scores is 0.005.

**Interpretation.** SHAP values attribute each prediction additively to its inputs [Lundberg & Lee, 2017]. Permutation importance on held-out data complements them by measuring the drop in performance.

---

## References

- Bergsma, W. (2013). A bias-correction for Cramér's V and Tschuprow's T. *Journal of the Korean Statistical Society*, 42(3), 323–328. https://doi.org/10.1016/j.jkss.2012.10.002
- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13, 281–305. https://jmlr.org/papers/v13/bergstra12a.html
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324
- Elkan, C. (2001). The foundations of cost-sensitive learning. *Proceedings of the 17th International Joint Conference on Artificial Intelligence*, 973–978. https://cseweb.ucsd.edu/~elkan/rescale.pdf
- Hand, D. J., & Henley, W. E. (1997). Statistical classification methods in consumer credit scoring: A review. *Journal of the Royal Statistical Society: Series A*, 160(3), 523–541. https://doi.org/10.1111/j.1467-985X.1997.00078.x
- Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), Article 15. https://doi.org/10.1145/2382577.2382579
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems 30*, 3146–3154. https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree
- Lessmann, S., Baesens, B., Seow, H.-V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. *European Journal of Operational Research*, 247(1), 124–136. https://doi.org/10.1016/j.ejor.2015.05.030
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems 30*. https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions
- Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22–31. https://doi.org/10.1016/j.dss.2014.03.001
- Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *Proceedings of the 22nd International Conference on Machine Learning*, 625–632. https://doi.org/10.1145/1102351.1102430
- O'Brien, R. M. (2007). A caution regarding rules of thumb for variance inflation factors. *Quality & Quantity*, 41(5), 673–690. https://doi.org/10.1007/s11135-006-9018-6
- Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., & Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929. https://doi.org/10.1111/ecog.02881
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
- Siddiqi, N. (2006). *Credit risk scorecards: Developing and implementing intelligent credit scoring*. Wiley. https://doi.org/10.1002/9781119201731
- van den Goorbergh, R., van Smeden, M., Timmerman, D., & Van Calster, B. (2022). The harm of class imbalance corrections for risk prediction models: Illustration and simulation using logistic regression. *Journal of the American Medical Informatics Association*, 29(9), 1525–1534. https://doi.org/10.1093/jamia/ocac093
- Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012). New insights into churn prediction in the telecommunication sector: A profit driven data mining approach. *European Journal of Operational Research*, 218(1), 211–229. https://doi.org/10.1016/j.ejor.2011.09.031
- Yurdakul, B., & Naranjo, J. (2020). Statistical properties of the population stability index. *The Journal of Risk Model Validation*, 14(4), 89–100. https://doi.org/10.21314/JRMV.2020.227
