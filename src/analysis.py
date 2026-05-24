import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, Tuple

class AnalysisEngine:
    def __init__(self, processed_df: pd.DataFrame, predictions_df: pd.DataFrame):
        self.processed_df = processed_df
        self.predictions_df = predictions_df
        self.analysis_results = {}

    def generate_risk_distribution_report(self) -> Dict:
        report = {}

        # Rule-based distribution
        rule_dist = self.predictions_df["rule_based_risk"].value_counts()
        report["rule_based"] = {
            level: {
                "count": int(count),
                "percentage": float(count / len(self.predictions_df) * 100),
            }
            for level, count in rule_dist.items()
        }

        # ML-based distribution
        ml_dist = self.predictions_df["ml_based_risk"].value_counts()
        report["ml_based"] = {
            level: {
                "count": int(count),
                "percentage": float(count / len(self.predictions_df) * 100),
            }
            for level, count in ml_dist.items()
        }

        return report

    def generate_demographic_analysis(self) -> Dict:
        analysis = {}

        # By age group
        age_bins = [0, 6, 12, 24, 60]
        age_labels = ["0-6mo", "6-12mo", "12-24mo", "24-60mo"]
        self.processed_df["age_group"] = pd.cut(
            self.processed_df["age_months"], bins=age_bins, labels=age_labels, right=False
        )

        age_risk = pd.crosstab(
            self.processed_df["age_group"],
            self.predictions_df["rule_based_risk"],
            normalize="index",
        ) * 100

        analysis["by_age_group"] = age_risk.to_dict()

        # By gender
        gender_map = {"M": "Male", "F": "Female"}
        self.processed_df["gender_name"] = self.processed_df["gender"].map(gender_map)

        gender_risk = pd.crosstab(
            self.processed_df["gender_name"],
            self.predictions_df["rule_based_risk"],
            normalize="index",
        ) * 100

        analysis["by_gender"] = gender_risk.to_dict()

        return analysis

    def generate_feature_correlation_analysis(self) -> Dict:
        analysis = {}

        # Map risk to numeric
        risk_map = {"low": 0, "medium": 1, "high": 2}
        risk_numeric = self.predictions_df["rule_based_risk"].map(risk_map)

        # Calculate correlations
        features_to_analyze = [
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

        correlations = {}
        for feature in features_to_analyze:
            if feature in self.processed_df.columns:
                corr = self.processed_df[feature].corr(risk_numeric)
                correlations[feature] = float(corr)

        analysis["feature_correlations"] = dict(
            sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        return analysis

    def generate_classification_metrics(self) -> Dict:
        metrics = {}

        # Map predictions to numeric
        rule_map = {"low": 0, "medium": 1, "high": 2}
        ml_map = {"low": 0, "medium": 1, "high": 2}

        y_true = self.processed_df["risk_label"].map(rule_map)
        y_pred_rule = self.predictions_df["rule_based_risk"].map(rule_map)
        y_pred_ml = self.predictions_df["ml_based_risk"].map(ml_map)

        # Rule-based metrics
        metrics["rule_based"] = {
            "confusion_matrix": confusion_matrix(y_true, y_pred_rule).tolist(),
            "classification_report": classification_report(
                y_true, y_pred_rule, target_names=["Low", "Medium", "High"], output_dict=True
            ),
        }

        # ML-based metrics
        metrics["ml_based"] = {
            "confusion_matrix": confusion_matrix(y_true, y_pred_ml).tolist(),
            "classification_report": classification_report(
                y_true, y_pred_ml, target_names=["Low", "Medium", "High"], output_dict=True
            ),
        }

        return metrics

    def generate_risk_factor_analysis(self) -> Dict:
        analysis = {}

        high_risk_mask = self.predictions_df["rule_based_risk"] == "high"
        low_risk_mask = self.predictions_df["rule_based_risk"] == "low"

        high_risk_df = self.processed_df[high_risk_mask]
        low_risk_df = self.processed_df[low_risk_mask]

        # Compare means
        comparison = {}
        for feature in [
            "age_months",
            "weight_kg",
            "height_cm",
            "bmi",
            "haz",
            "waz",
            "whz",
            "energy_pct_required",
            "protein_g_per_kg",
        ]:
            if feature in self.processed_df.columns:
                comparison[feature] = {
                    "high_risk_mean": float(high_risk_df[feature].mean()),
                    "low_risk_mean": float(low_risk_df[feature].mean()),
                    "difference": float(
                        high_risk_df[feature].mean() - low_risk_df[feature].mean()
                    ),
                }

        analysis["high_vs_low_risk"] = comparison

        return analysis

    def generate_full_report(self) -> Dict:
        print("\n" + "=" * 70)
        print("GENERATING COMPREHENSIVE ANALYSIS REPORT")
        print("=" * 70 + "\n")

        self.analysis_results = {
            "risk_distribution": self.generate_risk_distribution_report(),
            "demographic_analysis": self.generate_demographic_analysis(),
            "feature_correlations": self.generate_feature_correlation_analysis(),
            "classification_metrics": self.generate_classification_metrics(),
            "risk_factor_analysis": self.generate_risk_factor_analysis(),
        }

        self._print_report()
        return self.analysis_results

    def _print_report(self) -> None:
        print("1. Risk Distribution")
        print("-" * 70)
        print("\nRule-Based System:")
        for level, data in self.analysis_results["risk_distribution"]["rule_based"].items():
            print(
                f"  {level.upper():>10s}: {data['count']:>6d} ({data['percentage']:>5.1f}%)"
            )

        print("\nML-Based System:")
        for level, data in self.analysis_results["risk_distribution"]["ml_based"].items():
            print(
                f"  {level.upper():>10s}: {data['count']:>6d} ({data['percentage']:>5.1f}%)"
            )

        print("\n2. Feature Correlations with Risk")
        print("-" * 70)
        for feature, corr in self.analysis_results["feature_correlations"][
            "feature_correlations"
        ].items():
            bar_length = int(abs(corr) * 30)
            bar = "█" * bar_length if corr > 0 else "▓" * bar_length
            print(f"  {feature:25s} {corr:>7.4f} {bar}")

        print("\n3. High-Risk vs Low-Risk Comparison")
        print("-" * 70)
        print(f"{'Feature':<25s} {'High-Risk Mean':>15s} {'Low-Risk Mean':>15s} {'Difference':>15s}")
        print("-" * 70)
        for feature, data in self.analysis_results["risk_factor_analysis"][
            "high_vs_low_risk"
        ].items():
            print(
                f"{feature:<25s} {data['high_risk_mean']:>15.2f} {data['low_risk_mean']:>15.2f} {data['difference']:>15.2f}"
            )

    def save_report(self, output_path: str) -> None:
        import json

        with open(output_path, "w") as f:
            json.dump(self.analysis_results, f, indent=2)
        print(f"\nAnalysis report saved to {output_path}")


class VisualizationEngine:
    def __init__(self, processed_df: pd.DataFrame, predictions_df: pd.DataFrame):
        self.processed_df = processed_df
        self.predictions_df = predictions_df
        self.output_dir = Path(__file__).parent.parent / "visualizations"
        self.output_dir.mkdir(exist_ok=True)

    def plot_risk_distribution(self) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Rule-based
        rule_counts = self.predictions_df["rule_based_risk"].value_counts()
        axes[0].bar(rule_counts.index, rule_counts.values, color=["green", "orange", "red"])
        axes[0].set_title("Rule-Based System Risk Distribution", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Count")
        axes[0].set_xlabel("Risk Level")

        # ML-based
        ml_counts = self.predictions_df["ml_based_risk"].value_counts()
        axes[1].bar(ml_counts.index, ml_counts.values, color=["green", "orange", "red"])
        axes[1].set_title("ML-Based System Risk Distribution", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Count")
        axes[1].set_xlabel("Risk Level")

        plt.tight_layout()
        plt.savefig(self.output_dir / "01_risk_distribution.png", dpi=300, bbox_inches="tight")
        print(f"Saved: 01_risk_distribution.png")
        plt.close()

    def plot_zscore_distribution(self) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for idx, zscore in enumerate(["haz", "waz", "whz"]):
            for risk in ["low", "medium", "high"]:
                mask = self.predictions_df["rule_based_risk"] == risk
                data = self.processed_df[mask][zscore]
                axes[idx].hist(data, alpha=0.5, label=risk.capitalize(), bins=30)

            axes[idx].set_title(f"{zscore.upper()} Distribution by Risk", fontweight="bold")
            axes[idx].set_xlabel("Z-Score")
            axes[idx].set_ylabel("Frequency")
            axes[idx].legend()
            axes[idx].axvline(x=-2, color="red", linestyle="--", alpha=0.5, label="Malnutrition Threshold")

        plt.tight_layout()
        plt.savefig(self.output_dir / "02_zscore_distributions.png", dpi=300, bbox_inches="tight")
        print(f"Saved: 02_zscore_distributions.png")
        plt.close()

    def plot_nutrition_indicators(self) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Energy
        for risk in ["low", "medium", "high"]:
            mask = self.predictions_df["rule_based_risk"] == risk
            data = self.processed_df[mask]["energy_pct_required"]
            axes[0].hist(data, alpha=0.5, label=risk.capitalize(), bins=30)

        axes[0].set_title("Energy Intake (% of Requirement) by Risk", fontweight="bold")
        axes[0].set_xlabel("Energy %")
        axes[0].set_ylabel("Frequency")
        axes[0].axvline(x=90, color="green", linestyle="--", alpha=0.7, label="Adequate Level")
        axes[0].axvline(x=70, color="red", linestyle="--", alpha=0.7, label="Deficiency Level")
        axes[0].legend()

        # Protein
        for risk in ["low", "medium", "high"]:
            mask = self.predictions_df["rule_based_risk"] == risk
            data = self.processed_df[mask]["protein_g_per_kg"]
            axes[1].hist(data, alpha=0.5, label=risk.capitalize(), bins=30)

        axes[1].set_title("Protein Intake (g/kg) by Risk", fontweight="bold")
        axes[1].set_xlabel("Protein (g/kg)")
        axes[1].set_ylabel("Frequency")
        axes[1].axvline(x=1.1, color="green", linestyle="--", alpha=0.7, label="Adequate Level")
        axes[1].axvline(x=0.9, color="red", linestyle="--", alpha=0.7, label="Deficiency Level")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "03_nutrition_indicators.png", dpi=300, bbox_inches="tight")
        print(f"Saved: 03_nutrition_indicators.png")
        plt.close()

    def plot_demographic_analysis(self) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # By age group
        age_bins = [0, 6, 12, 24, 60]
        age_labels = ["0-6mo", "6-12mo", "12-24mo", "24-60mo"]
        self.processed_df["age_group"] = pd.cut(
            self.processed_df["age_months"], bins=age_bins, labels=age_labels, right=False
        )

        age_risk = pd.crosstab(
            self.processed_df["age_group"], self.predictions_df["rule_based_risk"]
        )
        age_risk.plot(kind="bar", ax=axes[0], color=["green", "orange", "red"])
        axes[0].set_title("Risk Distribution by Age Group", fontweight="bold")
        axes[0].set_xlabel("Age Group")
        axes[0].set_ylabel("Count")
        axes[0].legend(title="Risk Level")
        axes[0].tick_params(axis="x", rotation=45)

        # By gender
        gender_map = {"M": "Male", "F": "Female"}
        self.processed_df["gender_name"] = self.processed_df["gender"].map(gender_map)

        gender_risk = pd.crosstab(
            self.processed_df["gender_name"], self.predictions_df["rule_based_risk"]
        )
        gender_risk.plot(kind="bar", ax=axes[1], color=["green", "orange", "red"])
        axes[1].set_title("Risk Distribution by Gender", fontweight="bold")
        axes[1].set_xlabel("Gender")
        axes[1].set_ylabel("Count")
        axes[1].legend(title="Risk Level")
        axes[1].tick_params(axis="x", rotation=0)

        plt.tight_layout()
        plt.savefig(self.output_dir / "04_demographic_analysis.png", dpi=300, bbox_inches="tight")
        print(f"Saved: 04_demographic_analysis.png")
        plt.close()

    def generate_all_visualizations(self) -> None:
        print("\n" + "=" * 70)
        print("Visualize")
        print("=" * 70 + "\n")

        self.plot_risk_distribution()
        self.plot_zscore_distribution()
        self.plot_nutrition_indicators()
        self.plot_demographic_analysis()

        print(f"\nAll visualizations saved to {self.output_dir}")
