"""Load the pipe-delimited campaign file and give the coded columns readable names."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Readable names taken from the client's data dictionary. The original codes are kept
# in reports so every result can be traced back to the source file.
COLUMN_NAMES = {
    "Feature_ae_0": "age",
    "Feature_dn_1": "call_duration_s",
    "Feature_cn_2": "call_attempts",
    "Feature_ps_3": "days_since_prev_campaign",
    "Feature_ps_4": "prev_campaign_attempts",
    "Feature_ee_5": "employment_variation_rate",
    "Feature_cx_6": "consumer_price_index",
    "Feature_cx_7": "consumer_confidence_index",
    "Feature_em_8": "interbank_rate_3m",
    "Feature_nd_9": "number_of_employees",
    "Feature_jd_10": "employment_type",
    "Feature_md_11": "civil_status",
    "Feature_ed_12": "education",
    "Feature_dd_13": "credit_in_default",
    "Feature_hd_14": "has_home_loan",
    "Feature_ld_15": "has_personal_loan",
    "Feature_cd_16": "contact_medium",
    "Feature_md_17": "last_contact_month",
    "Feature_dd_18": "last_contact_weekday",
    "Feature_pd_19": "previous_outcome",
    "Response": "response",
}
CODE_OF = {v: k for k, v in COLUMN_NAMES.items()}

NUMERIC_RAW = [
    "age", "call_duration_s", "call_attempts", "days_since_prev_campaign", "prev_campaign_attempts",
    "employment_variation_rate", "consumer_price_index", "consumer_confidence_index",
    "interbank_rate_3m", "number_of_employees",
]
CATEGORICAL_RAW = [
    "employment_type", "civil_status", "education", "credit_in_default", "has_home_loan",
    "has_personal_loan", "contact_medium", "last_contact_month", "last_contact_weekday", "previous_outcome",
]

LABELS = {
    "age": "Age",
    "call_duration_s": "Call duration (s)",
    "call_attempts": "Call attempts (this campaign)",
    "days_since_prev_campaign": "Days since previous campaign",
    "prev_campaign_attempts": "Call attempts (previous campaign)",
    "employment_variation_rate": "Employment variation rate",
    "consumer_price_index": "Consumer price index",
    "consumer_confidence_index": "Consumer confidence index",
    "interbank_rate_3m": "3-month interbank rate",
    "number_of_employees": "Number of employees",
    "employment_type": "Employment type",
    "civil_status": "Civil status",
    "education": "Education",
    "credit_in_default": "Credit in default",
    "has_home_loan": "Has home loan",
    "has_personal_loan": "Has personal loan",
    "contact_medium": "Contact medium",
    "last_contact_month": "Last contact month",
    "last_contact_weekday": "Last contact weekday",
    "previous_outcome": "Previous campaign outcome",
    "prev_contact_recency": "Previous contact recency",
    "previously_contacted": "Previously contacted",
}


def load_raw(path: Path, sep: str = "|") -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip() for c in df.columns]
    missing = set(COLUMN_NAMES) - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected file layout, missing columns: {sorted(missing)}")
    return df.rename(columns=COLUMN_NAMES)


def label(name: str) -> str:
    return LABELS.get(name, name.replace("_", " ").capitalize())
