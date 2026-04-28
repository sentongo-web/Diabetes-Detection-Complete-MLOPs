"""
src/data/loader.py
------------------
Reads the raw Stata (RIF_final.dta) and CSV datasets from Nakaseke Hospital
and returns them as plain pandas DataFrames.

Why two datasets?
  _ RIF_final.dta  - The original field-collected NCD survey (3 471 patients,
    287 variables). This is the authoritative clinical source.
  _ diabetes_dataset.csv - A pre-processed supplementary sheet (2 004 patients)
    that adds validated diabetes labels from a parallel extraction.

We keep raw loading separate from cleaning so that any analyst can import just
this module to see the original data before any transformation.
"""

import pandas as pd
from pathlib import Path
import warnings
import logging

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def load_rif(path: str | Path = "data/raw/RIF_final.dta") -> pd.DataFrame:
    """
    Load the Stata Research Instrument File (RIF).

    The file was exported from REDCap / SurveyCTO by Nakaseke Hospital
    research staff. All variable labels are kept via pyreadstat under the
    hood; pandas.read_stata exposes them as category dtypes.

    Returns
    -------
    pd.DataFrame  - raw, unprocessed, exactly as stored in the .dta file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"RIF file not found at {path}. "
            "Place RIF_final.dta inside data/raw/ and retry."
        )
    logger.info("Loading RIF dataset from %s _", path)
    df = pd.read_stata(path, convert_categoricals=True)
    logger.info("RIF loaded: %d rows x %d columns", *df.shape)
    return df


def load_csv(path: str | Path = "data/raw/diabetes_dataset.csv") -> pd.DataFrame:
    """
    Load the supplementary diabetes CSV.

    This sheet was produced by the hospital's data-management team and
    contains a cleaner subset of variables alongside a validated diabetes
    binary label (1 = diabetic, 0 = not diabetic).

    Returns
    -------
    pd.DataFrame  - raw CSV without modification.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found at {path}. "
            "Place diabetes_dataset.csv inside data/raw/ and retry."
        )
    logger.info("Loading CSV dataset from %s _", path)
    df = pd.read_csv(path)
    logger.info("CSV loaded: %d rows x %d columns", *df.shape)
    return df


def load_all(rif_path: str | Path = "data/raw/RIF_final.dta",
             csv_path: str | Path = "data/raw/diabetes_dataset.csv"
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper: returns (rif_df, csv_df) in one call.
    """
    return load_rif(rif_path), load_csv(csv_path)
