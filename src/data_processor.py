import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict, List
from constants import (
    WHO_THRESHOLDS,
    ENERGY_THRESHOLDS,
    PROTEIN_THRESHOLDS,
    AGE_GROUPS,
    ENERGY_REQUIREMENTS,
    PROTEIN_REQUIREMENTS,
    RISK_LEVELS,
)
from risk_classifier import RuleBasedClassifier
from zscore_calculator import calculate_all_zscores, get_calculator

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df = None
        self.processed_df = None
        self.rule_classifier = RuleBasedClassifier()

    def load_data(self) -> pd.DataFrame:
        try:
            self.df = pd.read_csv(self.data_path, encoding="utf-8")
            print(f"Loaded {len(self.df)} records from {self.data_path.name}")
            return self.df
        except Exception as e:
            raise ValueError(f"Error loading data: {e}")

    def standardize_columns(self) -> pd.DataFrame:
        column_mapping = {
            "Jenis Kelamin": "gender",
            "Umur (bulan)": "age_months",
            "Tinggi Badan (cm)": "height_cm",
            "Berat Badan (kg)": "weight_kg",
            "Stunting": "stunting_status",
            "Wasting": "wasting_status",
        }

        self.df = self.df.rename(columns=column_mapping)

        # Standardize gender values
        self.df["gender"] = self.df["gender"].str.strip().str.lower()
        self.df["gender"] = self.df["gender"].map(
            {"laki-laki": "M", "perempuan": "F"}
        )

        # Ensure numeric columns
        self.df["age_months"] = pd.to_numeric(self.df["age_months"], errors="coerce")
        self.df["height_cm"] = pd.to_numeric(self.df["height_cm"], errors="coerce")
        self.df["weight_kg"] = pd.to_numeric(self.df["weight_kg"], errors="coerce")

        # Remove rows with missing critical values
        self.df = self.df.dropna(
            subset=["age_months", "height_cm", "weight_kg", "gender"]
        )

        print(f"Standardized columns. Remaining records: {len(self.df)}")
        return self.df

    def calculate_bmi(self) -> pd.DataFrame:
        self.df["bmi"] = self.df["weight_kg"] / (
            (self.df["height_cm"] / 100) ** 2
        )
        return self.df

    def get_age_group(self, age_months: float) -> str:
        for group, (min_age, max_age) in AGE_GROUPS.items():
            if min_age <= age_months < max_age:
                return group
        return "24-60"  # Default to oldest group

    def calculate_energy_requirement(self, age_months: float) -> float:
        age_group = self.get_age_group(age_months)
        return ENERGY_REQUIREMENTS.get(age_group, 1400)

    def calculate_protein_requirement(self, age_months: float) -> float:
        age_group = self.get_age_group(age_months)
        return PROTEIN_REQUIREMENTS.get(age_group, 20)

    def add_nutrition_features(
        self,
        energy_pct_required: List[float] = None,
        protein_g_per_kg: List[float] = None,
    ) -> pd.DataFrame:
        if energy_pct_required is None:
            # Generate realistic scenarios: 60%, 80%, 95%, 110%
            if "Energy_kcal" in self.df.columns:
                energy_pct_required = (
                    self.df["Energy_kcal"]
                    / self.df["age_months"].apply(self.calculate_energy_requirement)
                    * 100
                )
            else:
                np.random.seed(42)
                energy_pct_required = np.random.choice(
                    [60, 75, 85, 95, 110], size=len(self.df)
                )

        if protein_g_per_kg is None:
            # Generate realistic protein intake scenarios
            if "Protein_g" in self.df.columns:
                protein_g_per_kg = self.df["Protein_g"] / self.df["weight_kg"]
            else:
                np.random.seed(42)
                protein_g_per_kg = np.random.normal(1.0, 0.3, size=len(self.df))
                protein_g_per_kg = np.clip(protein_g_per_kg, 0.5, 1.5)

        self.df["energy_pct_required"] = energy_pct_required
        self.df["protein_g_per_kg"] = protein_g_per_kg

        # Calculate absolute values
        self.df["daily_energy_kcal"] = self.df["age_months"].apply(
            self.calculate_energy_requirement
        )
        self.df["daily_protein_g"] = self.df["age_months"].apply(
            self.calculate_protein_requirement
        )

        return self.df

    def create_zscore_features(self) -> pd.DataFrame:
        logger.info("Calculating Z-scores using WHO reference data...")
        
        # Initialize the calculator
        calculator = get_calculator()
        
        # Calculate Z-scores for each row using actual measurements
        def calc_zscores(row):
            try:
                haz, waz, whz = calculate_all_zscores(
                    age_months=row["age_months"],
                    weight_kg=row["weight_kg"],
                    height_cm=row["height_cm"],
                    gender=row["gender"]
                )
                return pd.Series({"haz": haz, "waz": waz, "whz": whz})
            except Exception as e:
                logger.warning(f"Error calculating Z-scores for row: {e}")
                return pd.Series({"haz": 0.0, "waz": 0.0, "whz": 0.0})
        
        zscore_df = self.df.apply(calc_zscores, axis=1)
        self.df["haz"] = zscore_df["haz"]
        self.df["waz"] = zscore_df["waz"]
        self.df["whz"] = zscore_df["whz"]
        
        logger.info(f"Z-scores calculated for {len(self.df)} records")
        return self.df

    def classify_risk_level(self, row: pd.Series) -> str:
        risk, _ = self.rule_classifier.classify(
            age_months=row["age_months"],
            weight_kg=row["weight_kg"],
            height_cm=row["height_cm"],
            haz=row["haz"],
            waz=row["waz"],
            whz=row["whz"],
            energy_pct=row["energy_pct_required"],
            protein_g_per_kg=row["protein_g_per_kg"],
        )
        return risk

    def create_risk_labels(self) -> pd.DataFrame:
        self.df["risk_label"] = self.df.apply(self.classify_risk_level, axis=1)
        return self.df

    def process(
        self,
        energy_pct_required: List[float] = None,
        protein_g_per_kg: List[float] = None,
    ) -> pd.DataFrame:
        self.load_data()
        self.standardize_columns()
        self.calculate_bmi()
        self.add_nutrition_features(energy_pct_required, protein_g_per_kg)
        self.create_zscore_features()
        self.create_risk_labels()

        self.processed_df = self.df[
            [
                "age_months",
                "gender",
                "weight_kg",
                "height_cm",
                "bmi",
                "haz",
                "waz",
                "whz",
                "energy_pct_required",
                "protein_g_per_kg",
                "daily_energy_kcal",
                "daily_protein_g",
                "risk_label",
            ]
        ].copy()

        print(f"✓ Processing complete. Final dataset: {len(self.processed_df)} records")
        print(f"\nRisk Distribution:\n{self.processed_df['risk_label'].value_counts()}")

        return self.processed_df

    def get_feature_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.processed_df is None:
            raise ValueError("Data not processed yet. Call process() first.")

        X = self.processed_df[
            [
                "age_months",
                "weight_kg",
                "height_cm",
                "bmi",
                "haz",
                "waz",
                "whz",
                "energy_pct_required",
                "protein_g_per_kg",
            ]
        ].values

        y = self.processed_df["risk_label"].map(RISK_LEVELS).values

        return X, y

    def save_processed_data(self, output_path: str) -> None:
        if self.processed_df is None:
            raise ValueError("No processed data to save.")

        self.processed_df.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")
