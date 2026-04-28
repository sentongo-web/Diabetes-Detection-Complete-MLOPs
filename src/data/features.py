"""
src/data/features.py
____________________
Feature engineering: transforms the clean tabular dataset into a richer
representation that gives tree-based and linear models better signal.

Three categories of features are added here:
  1. Clinical composites  _ medically motivated combinations (BMI categories,
                            metabolic risk score, waist-risk flags).
  2. Interaction terms    _ products of highly correlated predictors that
                            logistic regression and SVMs cannot capture alone.
  3. Polynomial terms     _ squared transformations for continuous variables
                            with known non-linear relationships to BP.

All new columns are clearly named so that SHAP explanations remain
interpretable by a clinician who did not write this code.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# _____________________________________________________________________________
# Clinical BMI category (WHO classification)
# _____________________________________________________________________________

def bmi_category(bmi: pd.Series) -> pd.Series:
    """
    0 = Underweight (<18.5),  1 = Normal (18.5-24.9),
    2 = Overweight (25-29.9), 3 = Obese (_30).
    """
    cats = pd.cut(
        bmi,
        bins=[-np.inf, 18.5, 25.0, 30.0, np.inf],
        labels=[0, 1, 2, 3],
        right=False,
    ).astype(float)
    return cats


def waist_risk_flag(waist: pd.Series, female: pd.Series) -> pd.Series:
    """
    WHO abdominal obesity thresholds:
      Women _ waist > 88 cm = high risk (1), else 0
      Men   _ waist > 102 cm = high risk (1), else 0
    """
    risk = pd.Series(0, index=waist.index, dtype=int)
    risk[(female == 1) & (waist > 88)]  = 1
    risk[(female == 0) & (waist > 102)] = 1
    return risk


def whr_risk_flag(whr: pd.Series, female: pd.Series) -> pd.Series:
    """
    WHO waist-hip ratio risk:
      Women _ WHR > 0.85 = high risk
      Men   _ WHR > 0.90 = high risk
    """
    risk = pd.Series(0, index=whr.index, dtype=int)
    risk[(female == 1) & (whr > 0.85)] = 1
    risk[(female == 0) & (whr > 0.90)] = 1
    return risk


def age_group(age: pd.Series) -> pd.Series:
    """
    Ordinal age band (known strong risk factor for hypertension):
    0 = <30,  1 = 30-44,  2 = 45-59,  3 = _60.
    """
    return pd.cut(
        age,
        bins=[-np.inf, 30, 45, 60, np.inf],
        labels=[0, 1, 2, 3],
        right=False,
    ).astype(float)


def lifestyle_risk_score(df: pd.DataFrame) -> pd.Series:
    """
    Composite lifestyle risk score (0-6):
    +1  current smoker
    +1  drinks > 14 units/week (WHO hazardous limit)
    +1  BMI _ 25 (overweight/obese)
    +1  low fruit+veg intake (combined servings < 0.2 normalised)
    +1  adds extra salt at the table
    +1  sedentary (no vigorous activity)
    """
    score = (
        df["current_smoker"].fillna(0).astype(int) +
        (df["drinks_per_week"].fillna(0) > 14).astype(int) +
        (df["bmi"].fillna(0) >= 25).astype(int) +
        ((df["fruit_servings_week"].fillna(0) + df["veg_servings_week"].fillna(0)) < 0.2).astype(int) +
        df["adds_salt"].fillna(0).astype(int) +
        (1 - df["vigorous_activity"].fillna(1).astype(int))
    )
    return score.clip(0, 6)


def metabolic_burden(df: pd.DataFrame) -> pd.Series:
    """
    Count of concurrent metabolic risk factors:
    Obesity + abdominal obesity + high WHR _ metabolic syndrome proxy.
    """
    burden = (
        (df["bmi"] >= 30).astype(int) +
        waist_risk_flag(df["waist_cm"], df["female"]) +
        whr_risk_flag(df["whr"], df["female"])
    )
    return burden.clip(0, 3)


# _____________________________________________________________________________
# Main engineering function
# _____________________________________________________________________________

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts the clean DataFrame from cleaner.clean_rif() and returns a new
    DataFrame with all original features plus the engineered ones.

    New columns added
    _________________
    bmi_cat            BMI WHO category (0-3)
    age_group          Ordinal age band (0-3)
    waist_risk         Abdominal obesity flag (0/1)
    whr_risk           Waist-hip ratio risk flag (0/1)
    lifestyle_score    Composite lifestyle risk (0-6)
    metabolic_burden   Metabolic syndrome proxy (0-3)
    age_bmi            Age _ BMI interaction
    age_whr            Age _ WHR interaction
    bmi_salt           BMI _ adds_salt interaction
    age_sq             Age squared (captures accelerating BP risk with age)
    bmi_sq             BMI squared
    """
    out = df.copy()

    out["bmi_cat"]          = bmi_category(out["bmi"])
    out["age_group"]        = age_group(out["age"])
    out["waist_risk"]       = waist_risk_flag(out["waist_cm"], out["female"])
    out["whr_risk"]         = whr_risk_flag(out["whr"], out["female"])
    out["lifestyle_score"]  = lifestyle_risk_score(out)
    out["metabolic_burden"] = metabolic_burden(out)

    # Interaction terms (standardise inputs first to avoid scale dominance)
    age_z   = (out["age"]   - out["age"].mean())   / (out["age"].std()   + 1e-8)
    bmi_z   = (out["bmi"]   - out["bmi"].mean())   / (out["bmi"].std()   + 1e-8)
    whr_z   = (out["whr"]   - out["whr"].mean())   / (out["whr"].std()   + 1e-8)

    out["age_bmi"]  = age_z * bmi_z
    out["age_whr"]  = age_z * whr_z
    out["bmi_salt"] = bmi_z * out["adds_salt"].fillna(0)

    # Polynomial terms
    out["age_sq"]   = age_z ** 2
    out["bmi_sq"]   = bmi_z ** 2

    n_new = out.shape[1] - df.shape[1]
    logger.info("engineer_features: added %d new columns; total=%d", n_new, out.shape[1])
    return out


def get_feature_names(df: pd.DataFrame, target: str = "hypertension") -> list[str]:
    """Return a sorted list of all feature column names (excluding the target)."""
    return [c for c in df.columns if c != target]
