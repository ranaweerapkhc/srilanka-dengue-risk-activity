"""Shared feature engineering for the Sri Lanka dengue-risk model.

Used by both the notebook (training) and the Streamlit app (inference), so the
model always sees features built exactly the same way.
"""
import numpy as np
import pandas as pd

# The raw file uses informal spellings. Map them to the official names used by
# the Epidemiology Unit. "Kalmunai" is a Regional Director of Health Services
# (RDHS) area inside Ampara district, which is why there are 26 areas, not 25.
DISTRICT_FIX = {
    "Anuradapura": "Anuradhapura",
    "Battocolo": "Batticaloa",
    "Kalminai": "Kalmunai",
    "Kegalla": "Kegalle",
    "Killinochchi": "Kilinochchi",
    "Mulative": "Mullaitivu",
    "N eliya": "Nuwara Eliya",
    "Puttalama": "Puttalam",
    "Rathnapura": "Ratnapura",
    "Vauniya": "Vavuniya",
}

# Raw inputs a user types into the front end
RAW_INPUTS = ["district", "month", "lag1", "lag2", "lag3", "lag12"]

NUMERIC_FEATURES = [
    "log_lag1", "log_lag2", "log_lag3", "log_lag12",
    "log_roll3", "growth_1m", "month_sin", "month_cos",
    "is_sw_monsoon", "is_ne_monsoon",
]
CATEGORICAL_FEATURES = ["district"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw inputs (district, month, last 3 months, same month last year)
    into model features. NaNs in lag12 are left for the imputer to handle."""
    out = pd.DataFrame(index=df.index)
    out["district"] = df["district"].astype(str)
    for c in ["lag1", "lag2", "lag3", "lag12"]:
        out[f"log_{c}"] = np.log1p(df[c].astype(float))  # counts are heavily right-skewed
    roll3 = df[["lag1", "lag2", "lag3"]].astype(float).mean(axis=1)
    out["log_roll3"] = np.log1p(roll3)
    # momentum: is the curve rising? (log ratio, +1 avoids divide-by-zero)
    out["growth_1m"] = np.log1p(df["lag1"].astype(float)) - np.log1p(df["lag2"].astype(float))
    m = df["month"].astype(int)
    out["month_sin"] = np.sin(2 * np.pi * m / 12)  # Dec and Jan are neighbours
    out["month_cos"] = np.cos(2 * np.pi * m / 12)
    # Sri Lanka's two monsoons drive mosquito breeding
    out["is_sw_monsoon"] = m.isin([5, 6, 7, 8, 9]).astype(int)     # Yala / south-west
    out["is_ne_monsoon"] = m.isin([12, 1, 2]).astype(int)         # Maha / north-east
    return out[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
