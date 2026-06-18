"""
Demo utilities for verosynthea-validator.

Provides pre-built model + data loaders for the canonical fairness
testing demonstration: a US-trained classifier tested against
Australian population data.
"""

import pickle
import pandas as pd
import numpy as np


# ── Schema mapping: AUSynth → UCI Adult ─────────────────────────

# AUSynth education → UCI Adult education mapping
_EDUCATION_MAP = {
    "Postgraduate Degree": "Doctorate",
    "Graduate Diploma and Graduate Certificate": "Masters",
    "Bachelor Degree": "Bachelors",
    "Advanced Diploma and Diploma": "Assoc-voc",
    "Certificate III & IV": "Some-college",
    "Certificate I & II": "HS-grad",
    "Certificate nfd": "HS-grad",
    "Year 12": "HS-grad",
    "Year 11": "11th",
    "Year 10": "10th",
    "Year 9 or below": "9th",
    "Inadequately described": "Some-college",
    "Not stated": "Some-college",
    "Not applicable": "Some-college",
}

# Education → education-num (ordinal)
_EDUCATION_NUM = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4,
    "9th": 5, "10th": 6, "11th": 7, "12th": 8, "HS-grad": 9,
    "Some-college": 10, "Assoc-voc": 11, "Assoc-acdm": 12,
    "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16,
}

# AUSynth occupation → UCI Adult occupation mapping
_OCCUPATION_MAP = {
    "Managers": "Exec-managerial",
    "Professionals": "Prof-specialty",
    "Technicians and Trades Workers": "Craft-repair",
    "Community and Personal Service Workers": "Other-service",
    "Clerical and Administrative Workers": "Adm-clerical",
    "Sales Workers": "Sales",
    "Machinery Operators and Drivers": "Transport-moving",
    "Labourers": "Handlers-cleaners",
    "Inadequately described": "Other-service",
    "Not stated": "Other-service",
    "Not applicable": "Other-service",
}

# AUSynth marital status → UCI Adult marital-status mapping
_MARITAL_MAP = {
    "Married": "Married-civ-spouse",
    "Separated": "Separated",
    "Divorced": "Divorced",
    "Widowed": "Widowed",
    "Never married": "Never-married",
    "Not stated": "Never-married",
    "Not applicable": "Never-married",
}

# AUSynth birthplace region → UCI Adult native-country best-fit
_COUNTRY_MAP = {
    "Australia": "United-States",  # Dominant local population
    "Oceania and Antarctica (excl. Australia)": "Outlying-US(Guam-USVI-etc)",
    "North-West Europe": "England",
    "Southern and Eastern Europe": "Italy",
    "North Africa and the Middle East": "Iran",
    "South-East Asia": "Vietnam",
    "North-East Asia": "China",
    "Southern and Central Asia": "India",
    "Americas": "United-States",
    "Sub-Saharan Africa": "South",
    "Inadequately described": "United-States",
    "Not stated": "United-States",
}

# AUSynth hours worked → midpoint numeric
_HOURS_MAP = {
    "None": 0, "1-15 hours": 8, "16-24 hours": 20,
    "25-34 hours": 30, "35-39 hours": 37, "40 hours": 40,
    "41-44 hours": 42, "45-49 hours": 47, "50-59 hours": 55,
    "60 hours or more": 65, "Not stated": 40,
    "Not applicable": 0,
}

# AUSynth age bracket → midpoint
_AGE_MAP = {
    "0-4 years": 2, "5-9 years": 7, "10-14 years": 12,
    "15-19 years": 17, "20-24 years": 22, "25-29 years": 27,
    "30-34 years": 32, "35-39 years": 37, "40-44 years": 42,
    "45-49 years": 47, "50-54 years": 52, "55-59 years": 57,
    "60-64 years": 62, "65-69 years": 67, "70-74 years": 72,
    "75-79 years": 77, "80-84 years": 82, "85-89 years": 87,
    "90-94 years": 92, "95-99 years": 97, "100 years and over": 100,
}

# AUSynth sex mapping
_SEX_MAP = {"Male": "Male", "Female": "Female"}


def map_ausynth_to_uci_adult(au_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map an AUSynth DataFrame to the UCI Adult Income schema so the
    US-trained baseline model can score it.

    Parameters
    ----------
    au_df : pd.DataFrame
        AUSynth data with columns like AGE5P, SEXP, INCP, HEAP,
        OCCP, MSTP, BPLP, HRWRP (standard AUSynth variable codes).

    Returns
    -------
    pd.DataFrame
        DataFrame with UCI Adult column names, ready for model.predict().

    Notes
    -----
    Unmapped features are set to reasonable defaults:
    - workclass: "Private" (most common UCI Adult value)
    - fnlwgt: median value (sampling weight, essentially noise)
    - capital-gain / capital-loss: 0 (not available in AUSynth)
    - relationship: derived from marital status + sex where possible
    - race: "White" (acknowledged simplification; this mismatch is
      part of why the model produces bias on AU data, and the demo
      highlights this)
    """
    n = len(au_df)
    uci = pd.DataFrame()

    # ── Map available features ──────────────────────────────────
    # Age
    if "AGE5P" in au_df.columns:
        uci["age"] = au_df["AGE5P"].map(_AGE_MAP).fillna(35).astype(int)
    elif "age" in au_df.columns:
        uci["age"] = au_df["age"]
    else:
        uci["age"] = 35

    # Sex
    if "SEXP" in au_df.columns:
        uci["sex"] = au_df["SEXP"].map(_SEX_MAP).fillna("Male")
    elif "sex" in au_df.columns:
        uci["sex"] = au_df["sex"]
    else:
        uci["sex"] = "Male"

    # Education
    if "HEAP" in au_df.columns:
        uci["education"] = au_df["HEAP"].map(_EDUCATION_MAP).fillna("Some-college")
    else:
        uci["education"] = "Some-college"
    uci["education-num"] = uci["education"].map(_EDUCATION_NUM).fillna(10).astype(int)

    # Marital status
    if "MSTP" in au_df.columns:
        uci["marital-status"] = au_df["MSTP"].map(_MARITAL_MAP).fillna("Never-married")
    else:
        uci["marital-status"] = "Never-married"

    # Occupation
    if "OCCP" in au_df.columns:
        uci["occupation"] = au_df["OCCP"].map(_OCCUPATION_MAP).fillna("Other-service")
    else:
        uci["occupation"] = "Other-service"

    # Hours per week
    if "HRWRP" in au_df.columns:
        uci["hours-per-week"] = au_df["HRWRP"].map(_HOURS_MAP).fillna(40).astype(int)
    else:
        uci["hours-per-week"] = 40

    # Country of birth
    if "BPLP" in au_df.columns:
        uci["native-country"] = au_df["BPLP"].map(_COUNTRY_MAP).fillna("United-States")
    else:
        uci["native-country"] = "United-States"

    # ── Set defaults for unmapped features ──────────────────────
    uci["workclass"] = "Private"
    uci["fnlwgt"] = 178356  # Median from UCI Adult
    uci["capital-gain"] = 0
    uci["capital-loss"] = 0

    # Relationship: derive from marital status + sex
    married = uci["marital-status"].isin(["Married-civ-spouse", "Married-AF-spouse"])
    male = uci["sex"] == "Male"
    uci["relationship"] = "Not-in-family"  # default
    uci.loc[married & male, "relationship"] = "Husband"
    uci.loc[married & ~male, "relationship"] = "Wife"

    # Race: acknowledged simplification
    uci["race"] = "White"

    # ── Build income target for fairness evaluation ─────────────
    # Map AUSynth weekly income to approximate annual, compare to $50K USD
    # This deliberately produces a distribution mismatch — that's the point
    if "INCP" in au_df.columns:
        uci["income_above_threshold"] = _map_income_target(au_df["INCP"])
    else:
        uci["income_above_threshold"] = 0

    # Preserve AUSynth demographic columns for fairness grouping
    for col in ["SEXP", "BPLP", "AGE5P", "OCCP", "profile_name", "profile_id"]:
        if col in au_df.columns:
            uci[col] = au_df[col].values

    # Reorder to match UCI Adult feature order
    feature_cols = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country",
    ]
    extra_cols = [c for c in uci.columns if c not in feature_cols]
    uci = uci[feature_cols + extra_cols]

    return uci


def _map_income_target(income_series: pd.Series) -> pd.Series:
    """
    Map AUSynth INCP (weekly income brackets) to a binary target
    comparable to UCI Adult's >$50K USD annual threshold.

    Uses approximate AUD weekly → annual conversion, then applies
    a rough AUD/USD adjustment. The mismatch between Australian
    and US income distributions is intentional — it's one of the
    bias sources the demo highlights.
    """
    # Weekly income midpoints (AUD)
    weekly_midpoints = {
        "Negative income": 0, "Nil income": 0,
        "$1-$149": 75, "$150-$299": 225, "$300-$399": 350,
        "$400-$499": 450, "$500-$649": 575, "$650-$799": 725,
        "$800-$999": 900, "$1,000-$1,249": 1125,
        "$1,250-$1,499": 1375, "$1,500-$1,749": 1625,
        "$1,750-$1,999": 1875, "$2,000-$2,999": 2500,
        "$3,000-$3,499": 3250, "$3,500 or more": 4000,
        "Not stated": 500, "Not applicable": 0,
    }
    # HF sample may include annual range in brackets, e.g.
    # "$1,500-$1,749 ($78,000-$90,999)" — strip the suffix
    cleaned = income_series.str.replace(r"\s*\(.*\)$", "", regex=True)
    weekly = cleaned.map(weekly_midpoints).fillna(500)
    annual_aud = weekly * 52
    annual_usd = annual_aud * 0.65  # rough AUD→USD
    return (annual_usd > 50000).astype(int)


def load_us_adult_baseline():
    """
    Load the pre-trained UCI Adult Income classifier from HuggingFace
    for fairness testing demonstrations.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted LogisticRegression pipeline expecting UCI Adult schema.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="verosynthea/us-adult-income-baseline",
        filename="model.pkl",
    )
    with open(path, "rb") as f:
        return pickle.load(f)


def load_ausynth_test_set(n: int = 5000) -> pd.DataFrame:
    """
    Load the AUSynth HuggingFace sample and map to UCI Adult schema
    for fairness testing.

    Parameters
    ----------
    n : int
        Number of rows to use (default 5000, the full sample).

    Returns
    -------
    pd.DataFrame
        DataFrame in UCI Adult schema with additional AUSynth
        demographic columns for fairness grouping.
    """
    from datasets import load_dataset

    ds = load_dataset("vero-synthea/ausynth-sample")
    df = ds["train"].to_pandas()
    if n < len(df):
        df = df.sample(n=n, random_state=42)
    return map_ausynth_to_uci_adult(df)
