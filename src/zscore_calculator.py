"""
Z-Score Calculator using WHO Child Growth Standards.

This module provides accurate Z-score calculations for:
- HAZ (Height/Length-for-Age Z-score)
- WAZ (Weight-for-Age Z-score)
- WHZ (Weight-for-Height Z-score) - approximated

Uses official WHO reference data from CSV files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class WHOZScoreCalculator:
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data3" / "z-scores who"
        else:
            data_dir = Path(data_dir)

        self.data_dir = data_dir
        self._lhfa_boys: Optional[pd.DataFrame] = None
        self._lhfa_girls: Optional[pd.DataFrame] = None
        self._wfa_boys: Optional[pd.DataFrame] = None
        self._wfa_girls: Optional[pd.DataFrame] = None
        self._loaded = False

    def _parse_european_decimal(self, value: str) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        return float(str(value).replace(",", "."))

    def _load_csv(self, filename: str) -> pd.DataFrame:
        filepath = self.data_dir / filename
        if not filepath.exists():
            logger.warning(f"WHO data file not found: {filepath}")
            return pd.DataFrame()

        df = pd.read_csv(filepath, sep=";", encoding="utf-8")
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        for col in df.columns:
            if col in ["Month", "c"]:  # Age column
                df[col] = df[col].astype(int)
            else:
                try:
                    df[col] = df[col].apply(self._parse_european_decimal)
                except (ValueError, AttributeError):
                    pass

        return df

    def load_data(self) -> bool:
        try:
            self._lhfa_boys = self._load_csv("lhfa_boys_0-to-5-years_zscores.csv")
            self._lhfa_girls = self._load_csv("lhfa_girls_0-to-5-years_zscores.csv")
            self._wfa_boys = self._load_csv("wfa_boys_0-to-5-years_zscores.csv")
            self._wfa_girls = self._load_csv("wfa_girls_0-to-5-years_zscores.csv")

            self._loaded = any([
                not self._lhfa_boys.empty,
                not self._lhfa_girls.empty,
                not self._wfa_boys.empty,
                not self._wfa_girls.empty,
            ])

            if self._loaded:
                logger.info("WHO reference data loaded successfully")
            else:
                logger.warning("No WHO reference data could be loaded")

            return self._loaded
        except Exception as e:
            logger.error(f"Error loading WHO data: {e}")
            return False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load_data()

    def _get_lms_for_age(
        self, df: pd.DataFrame, age_months: int, age_col: str = "Month"
    ) -> Tuple[float, float, float]:
        # Handle alternative column name for LHFA files
        if age_col not in df.columns and "c" in df.columns:
            age_col = "c"

        age_months = int(round(age_months))
        age_months = max(0, min(60, age_months)) 

        row = df[df[age_col] == age_months]
        if row.empty: # Find closest age
            ages = df[age_col].values
            closest_idx = np.argmin(np.abs(ages - age_months))
            row = df.iloc[[closest_idx]]

        L = row["L"].values[0]
        M = row["M"].values[0]
        S = row["S"].values[0]

        return L, M, S

    def _calculate_zscore_lms(
        self, value: float, L: float, M: float, S: float
    ) -> float:
        if M <= 0 or S <= 0:
            return 0.0

        if abs(L) < 0.0001:  # L approximately 0
            if value <= 0:
                return -4.0
            return np.log(value / M) / S
        else:
            if value <= 0:
                return -4.0
            return (((value / M) ** L) - 1) / (L * S)

    def calculate_haz(
        self, age_months: float, height_cm: float, gender: str
    ) -> float:
        self._ensure_loaded()

        gender = gender.upper()
        if gender in ["M", "MALE", "LAKI-LAKI"]:
            df = self._lhfa_boys
        else:
            df = self._lhfa_girls

        if df is None or df.empty:
            return self._fallback_haz(age_months, height_cm)

        try:
            L, M, S = self._get_lms_for_age(df, age_months)
            z = self._calculate_zscore_lms(height_cm, L, M, S)
            return float(np.clip(z, -4, 4))
        except Exception as e:
            logger.warning(f"Error calculating HAZ: {e}, using fallback")
            return self._fallback_haz(age_months, height_cm)

    def calculate_waz(
        self, age_months: float, weight_kg: float, gender: str
    ) -> float:
        self._ensure_loaded()

        gender = gender.upper()
        if gender in ["M", "MALE", "LAKI-LAKI"]:
            df = self._wfa_boys
        else:
            df = self._wfa_girls

        if df is None or df.empty:
            return self._fallback_waz(age_months, weight_kg)

        try:
            L, M, S = self._get_lms_for_age(df, age_months)
            z = self._calculate_zscore_lms(weight_kg, L, M, S)
            return float(np.clip(z, -4, 4))
        except Exception as e:
            logger.warning(f"Error calculating WAZ: {e}, using fallback")
            return self._fallback_waz(age_months, weight_kg)

    def calculate_whz(
        self, weight_kg: float, height_cm: float, gender: str = "M"
    ) -> float:
        if height_cm < 45:
            height_cm = 45
        elif height_cm > 120:
            height_cm = 120

        gender = gender.upper()
        if gender in ["M", "MALE", "LAKI-LAKI"]:
            expected_weight = 0.0062 * (height_cm ** 1.7)
            sd = 0.12 * expected_weight  # ~12% CV
        else:
            expected_weight = 0.0058 * (height_cm ** 1.7)
            sd = 0.11 * expected_weight  # ~11% CV

        if sd <= 0:
            sd = 1.0

        z = (weight_kg - expected_weight) / sd
        return float(np.clip(z, -4, 4))

    def _fallback_haz(self, age_months: float, height_cm: float) -> float:
        reference_heights = {
            0: 49.5, 3: 59.5, 6: 67.0, 12: 75.0,
            24: 87.0, 36: 95.0, 48: 102.0, 60: 109.0,
        }
        closest_age = min(reference_heights.keys(), key=lambda x: abs(x - age_months))
        ref_height = reference_heights[closest_age]
        std_dev = 3.0
        haz = (height_cm - ref_height) / std_dev
        return float(np.clip(haz, -4, 4))

    def _fallback_waz(self, age_months: float, weight_kg: float) -> float:
        reference_weights = {
            0: 3.3, 3: 6.4, 6: 7.9, 12: 9.6,
            24: 12.2, 36: 14.3, 48: 16.3, 60: 18.3,
        }
        closest_age = min(reference_weights.keys(), key=lambda x: abs(x - age_months))
        ref_weight = reference_weights[closest_age]
        std_dev = 1.2
        waz = (weight_kg - ref_weight) / std_dev
        return float(np.clip(waz, -4, 4))

    def calculate_all_zscores(
        self,
        age_months: float,
        weight_kg: float,
        height_cm: float,
        gender: str,
    ) -> Tuple[float, float, float]:
        haz = self.calculate_haz(age_months, height_cm, gender)
        waz = self.calculate_waz(age_months, weight_kg, gender)
        whz = self.calculate_whz(weight_kg, height_cm, gender)
        return haz, waz, whz


# Module-level singleton for convenience
_calculator: Optional[WHOZScoreCalculator] = None


def get_calculator() -> WHOZScoreCalculator:
    global _calculator
    if _calculator is None:
        _calculator = WHOZScoreCalculator()
        _calculator.load_data()
    return _calculator


def calculate_haz(age_months: float, height_cm: float, gender: str = "M") -> float:
    return get_calculator().calculate_haz(age_months, height_cm, gender)


def calculate_waz(age_months: float, weight_kg: float, gender: str = "M") -> float:
    return get_calculator().calculate_waz(age_months, weight_kg, gender)


def calculate_whz(weight_kg: float, height_cm: float, gender: str = "M") -> float:
    return get_calculator().calculate_whz(weight_kg, height_cm, gender)


def calculate_all_zscores(
    age_months: float, weight_kg: float, height_cm: float, gender: str = "M"
) -> Tuple[float, float, float]:
    return get_calculator().calculate_all_zscores(age_months, weight_kg, height_cm, gender)
