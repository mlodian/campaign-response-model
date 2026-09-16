# Data

## Source and handling

The model is trained on the client-supplied file `dataset.csv`:
- 35,000 rows and 21 columns (20 features and a binary `Response`);
- pipe (`|`) delimited;
- no missing cells.

The file is **not committed**. To reproduce the pipeline, copy it to `data/raw/dataset.csv`. All other files under `data/` are generated.

Category labels arrive partly masked (for example `Cat_9_t***n`). They are used exactly as supplied. The only interpretation made is that labels ending in `u***n` mean "unknown", which is treated as its own category.

## Data dictionary

The client's code, readable name and description for each field are below. The readable names are used in code and charts.

| Code | Name used | Type | Description (client dictionary) | Known before the call? |
|---|---|---|---|---|
| Feature_ae_0 | `age` | numeric | Customer age | yes |
| Feature_dn_1 | `call_duration_s` | numeric | Latest call duration in seconds, measured during the last sales call | **no: only known after the call; excluded** |
| Feature_cn_2 | `call_attempts` | numeric | Number of call attempts made to the customer | yes (attempts so far) |
| Feature_ps_3 | `days_since_prev_campaign` | numeric | Days since the last campaign (999 = no previous campaign) | yes |
| Feature_ps_4 | `prev_campaign_attempts` | numeric | Call attempts in the previous campaign | yes |
| Feature_ee_5 | `employment_variation_rate` | numeric | Employment variation rate (quarterly) | yes |
| Feature_cx_6 | `consumer_price_index` | numeric | Consumer price index (monthly) | yes |
| Feature_cx_7 | `consumer_confidence_index` | numeric | Consumer confidence index (monthly) | yes |
| Feature_em_8 | `interbank_rate_3m` | numeric | 3-month interbank rate (quarterly) | yes |
| Feature_nd_9 | `number_of_employees` | numeric | Number of employees (quarterly) | yes |
| Feature_jd_10 | `employment_type` | category | Type of employment | yes |
| Feature_md_11 | `civil_status` | category | Civil status | yes |
| Feature_ed_12 | `education` | category | Highest level of education | yes |
| Feature_dd_13 | `credit_in_default` | category | Has a credit facility in default | yes |
| Feature_hd_14 | `has_home_loan` | category | Has a home loan | yes |
| Feature_ld_15 | `has_personal_loan` | category | Has a personal loan | yes |
| Feature_cd_16 | `contact_medium` | category | Contact medium | yes |
| Feature_md_17 | `last_contact_month` | category | Month of last contact | yes (campaign timing) |
| Feature_dd_18 | `last_contact_weekday` | category | Weekday of last contact | yes |
| Feature_pd_19 | `previous_outcome` | category | Outcome of the previous campaign | yes |
| Response | `response` | binary | Took up the offer (1 = yes) | target |

## Issues found and how they are handled

| Issue | Evidence | Handling |
|---|---|---|
| Post-call leakage | Call duration is only known after the call. All 4 zero-second calls have response 0. Its Information Value is 1.93, the highest of any field. | Excluded from features. Kept only for a leakage benchmark. |
| Sentinel value | `days_since_prev_campaign = 999` (no previous campaign) on 96% of rows | Replaced by a recency band (`never`, 0–3, 4–6, 7–14, 15+ days). The raw column is not used: as a number, 999 would read as "contacted 999 days ago". |
| "Unknown" categories | For example, 21% of `credit_in_default` values are unknown | Kept as a category; unknown default status converts at less than half the rate of "no". |
| Very rare levels | `credit_in_default = yes` appears on 3 rows; one education level on 17 rows | Levels below 1% of training rows are grouped into "Other (rare)" (learned on training data). |
| Long tails | `call_attempts` runs up to 56 (99th percentile 14); `prev_campaign_attempts` up to 7 | Capped at the training-set 99th percentile inside the model pipeline |
| Duplicates | 11 rows identical across all 21 fields | Dropped; 34,989 rows remain |
| Redundant macro indicators | Employment variation rate, interbank rate and number of employees have Spearman correlations of 0.93–0.94; VIFs of 32–64 | Keep number of employees (highest IV); the other two are dropped |
| Redundant categories | "Previously contacted" duplicates the recency band (Cramér's V = 1.00) | Dropped |
| Class imbalance | 11.2% take-up | Kept as is. PR-AUC, lift and profit are used instead of accuracy, and the cut-off is set from economics. |
