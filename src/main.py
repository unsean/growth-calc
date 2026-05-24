from pathlib import Path
import numpy as np
import pandas as pd
from data_processor import DataProcessor
from risk_classifier import MLClassifier, RuleBasedClassifier, HybridClassifier

def main():
    print("\n" + "=" * 70)
    print("Malnutrition Risk Detection System")
    print("=" * 70 + "\n")

    print("Data Processing")
    print("-" * 70)

    data_path = Path(__file__).parent.parent / "data3" / "stunting_wasting_dataset.csv"

    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return

    processor = DataProcessor(str(data_path))
    processed_df = processor.process()

    output_path = Path(__file__).parent.parent / "data3" / "processed_data.csv"
    processor.save_processed_data(str(output_path))

    print("\nFeature Extraction")
    print("-" * 70)

    X, y = processor.get_feature_matrix()
    feature_names = [
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

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for risk_level, count in zip(unique, counts):
        risk_name = {0: "Low", 1: "Medium", 2: "High"}.get(risk_level, "Unknown")
        print(f"  - {risk_name}: {count} ({count/len(y)*100:.1f}%)")

    print("\nModel Training")
    print("-" * 70)

    ml_classifier = MLClassifier(max_depth=6, random_state=42)
    # metrics = ml_classifier.train(X, y, test_size=0.2)

    ml_classifier.print_metrics()
    ml_classifier.print_feature_importance(feature_names)

    print("\nRule-Based System Validation")
    print("-" * 70)

    rule_classifier = RuleBasedClassifier()

    # Test on sample records
    sample_indices = np.random.choice(len(processed_df), size=5, replace=False)
    print("\nSample Predictions (Rule-Based):")
    print("-" * 70)

    for idx in sample_indices:
        row = processed_df.iloc[idx]
        risk, explanation = rule_classifier.classify(
            age_months=row["age_months"],
            weight_kg=row["weight_kg"],
            height_cm=row["height_cm"],
            haz=row["haz"],
            waz=row["waz"],
            whz=row["whz"],
            energy_pct=row["energy_pct_required"],
            protein_g_per_kg=row["protein_g_per_kg"],
        )

        print(f"\nChild #{idx + 1}:")
        print(f"  Age: {row['age_months']:.0f}mo | Weight: {row['weight_kg']:.1f}kg | Height: {row['height_cm']:.1f}cm")
        print(f"  Z-Scores: HAZ={row['haz']:.2f}, WAZ={row['waz']:.2f}, WHZ={row['whz']:.2f}")
        print(f"  Nutrition: Energy={row['energy_pct_required']:.0f}%, Protein={row['protein_g_per_kg']:.2f}g/kg")
        print(f"  → Risk Level: {risk.upper()}")
        if explanation["severe_indicators"]:
            print(f"  Severe Indicators: {', '.join(explanation['severe_indicators'])}")
        if explanation["moderate_indicators"]:
            print(f"  Moderate Indicators: {', '.join(explanation['moderate_indicators'])}")
        if explanation["nutrition_indicators"]:
            print(f"  Nutrition Issues: {', '.join(explanation['nutrition_indicators'])}")

    print("\nHybrid Classification Examples")
    print("-" * 70)

    hybrid = HybridClassifier()

    print("\nDetailed Predictions (Hybrid Approach):")
    print("-" * 70)

    for idx in sample_indices[:3]:
        row = processed_df.iloc[idx]

        # Get prediction
        X_sample = X[idx].reshape(1, -1)
        ml_pred = ml_classifier.predict(X_sample)[0]

        # Hybrid classification
        result = hybrid.classify_hybrid(
            age_months=row["age_months"],
            weight_kg=row["weight_kg"],
            height_cm=row["height_cm"],
            haz=row["haz"],
            waz=row["waz"],
            whz=row["whz"],
            energy_pct=row["energy_pct_required"],
            protein_g_per_kg=row["protein_g_per_kg"],
            ml_prediction=ml_pred,
        )

        print(f"\n{'='*70}")
        print(f"CHILD #{idx + 1} ASSESSMENT")
        print(f"{'='*70}")
        print(f"Demographics: {row['age_months']:.0f} months old, {row['weight_kg']:.1f}kg, {row['height_cm']:.1f}cm")
        print(f"\nBiometric Indicators:")
        print(f"  Height-for-Age Z-score (HAZ):     {row['haz']:>7.2f}")
        print(f"  Weight-for-Age Z-score (WAZ):     {row['waz']:>7.2f}")
        print(f"  Weight-for-Height Z-score (WHZ):  {row['whz']:>7.2f}")
        print(f"  BMI:                               {row['bmi']:>7.2f}")
        print(f"\nNutrition Intake:")
        print(f"  Energy (% of requirement):         {row['energy_pct_required']:>6.0f}%")
        print(f"  Protein (g/kg body weight):        {row['protein_g_per_kg']:>6.2f}g/kg")
        print(f"\nClassification Results:")
        print(f"  Rule-Based System:  {result['rule_based_risk'].upper():>10s}")
        print(f"  ML-Based System:    {result['ml_based_risk'].upper() if result['ml_based_risk'] else 'N/A':>10s}")
        print(f"  Final Decision:     {result['final_risk'].upper():>10s}")
        print(f"\nRecommendation:")
        print(f"  {result['recommendation']}")

    print("\nSystem Statistics")
    print("-" * 70)

    # Compare rule-based vs ML on full dataset
    rule_predictions = []
    for idx in range(len(processed_df)):
        row = processed_df.iloc[idx]
        risk, _ = rule_classifier.classify(
            age_months=row["age_months"],
            weight_kg=row["weight_kg"],
            height_cm=row["height_cm"],
            haz=row["haz"],
            waz=row["waz"],
            whz=row["whz"],
            energy_pct=row["energy_pct_required"],
            protein_g_per_kg=row["protein_g_per_kg"],
        )
        rule_predictions.append(risk)

    ml_predictions = ml_classifier.predict(X)
    ml_predictions_str = [
        {0: "low", 1: "medium", 2: "high"}.get(p, "unknown") for p in ml_predictions
    ]

    print("\nRule-Based System Distribution:")
    rule_df = pd.Series(rule_predictions).value_counts()
    for risk, count in rule_df.items():
        print(f"  {risk.upper():>10s}: {count:>5d} ({count/len(rule_predictions)*100:>5.1f}%)")

    print("\nML-Based System Distribution:")
    ml_df = pd.Series(ml_predictions_str).value_counts()
    for risk, count in ml_df.items():
        print(f"  {risk.upper():>10s}: {count:>5d} ({count/len(ml_predictions_str)*100:>5.1f}%)")

    print("\nAgreement Analysis:")
    agreement = sum(
        1 for r, m in zip(rule_predictions, ml_predictions_str) if r == m
    )
    print(f"  Systems agree on: {agreement}/{len(rule_predictions)} ({agreement/len(rule_predictions)*100:.1f}%)")

    print("\nSave Result")
    print("-" * 70)

    # Create results dataframe
    results_df = processed_df.copy()
    results_df["rule_based_risk"] = rule_predictions
    results_df["ml_based_risk"] = ml_predictions_str

    results_path = Path(__file__).parent.parent / "data3" / "predictions.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved predictions to {results_path}")

    print("\n" + "=" * 70)
    print("Complete")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
