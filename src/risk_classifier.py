from typing import Dict, Tuple, Optional
import numpy as np
import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from constants import (
    RISK_LEVELS,
    RISK_LEVELS_REVERSE,
    WHO_THRESHOLDS,
    ENERGY_THRESHOLDS,
    PROTEIN_THRESHOLDS,
)

logger = logging.getLogger(__name__)

# Z-score thresholds (WHO standards)
SEVERE_THRESHOLD = WHO_THRESHOLDS["severe_malnutrition"]  # -3.0
MODERATE_THRESHOLD = WHO_THRESHOLDS["moderate_malnutrition"]  # -2.0

# Nutrition thresholds
ENERGY_ADEQUATE = ENERGY_THRESHOLDS["adequate"]  # 90%
ENERGY_BORDERLINE = ENERGY_THRESHOLDS["borderline"]  # 70%
PROTEIN_ADEQUATE = PROTEIN_THRESHOLDS["adequate"]  # 1.1 g/kg
PROTEIN_BORDERLINE = PROTEIN_THRESHOLDS["borderline"]  # 0.9 g/kg

class RuleBasedClassifier:
    """
    Rule-based classifier for malnutrition risk using WHO Z-score thresholds.
    
    Classification logic:
    - HIGH: Any severe Z-score (< -3.0) OR (moderate + nutrition issues)
    - MEDIUM: Moderate Z-score (-3.0 to -2.0) OR nutrition issues alone
    - LOW: All indicators normal
    """
    
    @staticmethod
    def validate_inputs(
        haz: float, waz: float, whz: float, 
        energy_pct: float, protein_g_per_kg: float
    ) -> bool:
        try:
            values = [haz, waz, whz, energy_pct, protein_g_per_kg]
            if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in values):
                return False
            if not all(-5 <= z <= 5 for z in [haz, waz, whz]):
                logger.warning(f"Z-scores out of typical range: HAZ={haz}, WAZ={waz}, WHZ={whz}")
            return True
        except (TypeError, ValueError):
            return False
    
    @staticmethod
    def classify(
        age_months: float,
        weight_kg: float,
        height_cm: float,
        haz: float,
        waz: float,
        whz: float,
        energy_pct: float,
        protein_g_per_kg: float,
    ) -> Tuple[str, Dict]:
        """
        Classify malnutrition risk using explicit WHO-based rules.

        Args:
            age_months: Child age in months (kept for future age-specific rules)
            weight_kg: Weight in kg (kept for future BMI-based rules)
            height_cm: Height in cm (kept for future height-specific rules)
            haz: Height-for-age Z-score
            waz: Weight-for-age Z-score
            whz: Weight-for-height Z-score
            energy_pct: Energy intake as % of requirement
            protein_g_per_kg: Protein intake in g/kg body weight

        Returns:
            Tuple of (risk_level, explanation_dict)
        """
        explanation = {
            "severe_indicators": [],
            "moderate_indicators": [],
            "nutrition_indicators": [],
        }


        if not RuleBasedClassifier.validate_inputs(haz, waz, whz, energy_pct, protein_g_per_kg):
            logger.warning("Invalid inputs provided, returning low risk with empty explanation")
            return "low", explanation
        
        # Severe malnutrition: Z-score < -3.0
        if haz < SEVERE_THRESHOLD:
            explanation["severe_indicators"].append(
                f"Severe stunting (HAZ {haz:.2f} < {SEVERE_THRESHOLD})"
            )
        if waz < SEVERE_THRESHOLD:
            explanation["severe_indicators"].append(
                f"Severe underweight (WAZ {waz:.2f} < {SEVERE_THRESHOLD})"
            )
        if whz < SEVERE_THRESHOLD:
            explanation["severe_indicators"].append(
                f"Severe wasting (WHZ {whz:.2f} < {SEVERE_THRESHOLD})"
            )

        # Moderate malnutrition: -3.0 <= Z-score < -2.0
        if SEVERE_THRESHOLD <= haz < MODERATE_THRESHOLD:
            explanation["moderate_indicators"].append(
                f"Moderate stunting (HAZ {haz:.2f})"
            )
        if SEVERE_THRESHOLD <= waz < MODERATE_THRESHOLD:
            explanation["moderate_indicators"].append(
                f"Moderate underweight (WAZ {waz:.2f})"
            )
        if SEVERE_THRESHOLD <= whz < MODERATE_THRESHOLD:
            explanation["moderate_indicators"].append(
                f"Moderate wasting (WHZ {whz:.2f})"
            )

        has_severe_anthro = len(explanation["severe_indicators"]) > 0
        has_moderate_anthro = len(explanation["moderate_indicators"]) > 0

        # Energy intake assessment
        if energy_pct < ENERGY_BORDERLINE:
            explanation["nutrition_indicators"].append(
                f"Low energy intake ({energy_pct:.0f}% < {ENERGY_BORDERLINE}% of requirement)"
            )
        elif energy_pct < ENERGY_ADEQUATE:
            explanation["nutrition_indicators"].append(
                f"Borderline energy intake ({energy_pct:.0f}% of requirement)"
            )

        # Protein intake assessment
        if protein_g_per_kg < PROTEIN_BORDERLINE:
            explanation["nutrition_indicators"].append(
                f"Low protein intake ({protein_g_per_kg:.2f} g/kg < {PROTEIN_BORDERLINE} g/kg) - may impair growth and immune function"
            )
        elif protein_g_per_kg < PROTEIN_ADEQUATE:
            explanation["nutrition_indicators"].append(
                f"Borderline protein intake ({protein_g_per_kg:.2f} g/kg) - consider increasing protein-rich foods"
            )

        # Excess energy intake assessment (obesity risk)
        if energy_pct >= 130:
            explanation["nutrition_indicators"].append(
                f"⚠️ Excessive energy intake ({energy_pct:.0f}% of requirement) - HIGH obesity risk if continued"
            )
        elif energy_pct >= 120:
            explanation["nutrition_indicators"].append(
                f"⚠️ High energy intake ({energy_pct:.0f}% of requirement) - obesity risk if continued"
            )
        elif energy_pct >= 110:
            explanation["nutrition_indicators"].append(
                f"Slightly elevated energy intake ({energy_pct:.0f}% of requirement) - monitor weight trend"
            )

        # Overweight/obesity indicators based on WHZ
        if whz >= 3.0:
            explanation["severe_indicators"].append(
                f"Obesity detected (WHZ {whz:.2f} ≥ 3.0) - immediate dietary intervention needed"
            )
        elif whz >= 2.0:
            explanation["moderate_indicators"].append(
                f"Overweight detected (WHZ {whz:.2f} ≥ 2.0) - risk of childhood obesity"
            )

        has_nutrition_issue = len(explanation["nutrition_indicators"]) > 0

        # 1) Any severe anthropometric indicator -> HIGH risk
        if has_severe_anthro:
            return "high", explanation

        # 2) Combination of moderate anthropometry + nutrition issues -> HIGH
        if has_moderate_anthro and has_nutrition_issue:
            return "high", explanation

        # 3) Either moderate anthropometry OR nutrition issues -> MEDIUM
        if has_moderate_anthro or has_nutrition_issue:
            return "medium", explanation

        # 4) No concerning indicators -> LOW
        return "low", explanation


class MLClassifier:
    def __init__(self, max_depth: int = 5, random_state: int = 42):
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state,
            min_samples_split=5,
            min_samples_leaf=2,
        )
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.metrics = {}

    def train(
        self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2
    ) -> Dict:
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Train dan evaluate
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self._calculate_metrics()

        print("Model training complete")
        return self.metrics

    def _calculate_metrics(self) -> None:
        self.metrics = {
            "accuracy": accuracy_score(self.y_test, self.y_pred),
            "precision": precision_score(
                self.y_test, self.y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                self.y_test, self.y_pred, average="weighted", zero_division=0
            ),
            "f1": f1_score(
                self.y_test, self.y_pred, average="weighted", zero_division=0
            ),
            "confusion_matrix": confusion_matrix(self.y_test, self.y_pred),
            "classification_report": classification_report(
                self.y_test, self.y_pred, target_names=["Low", "Medium", "High"]
            ),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importance(self, feature_names: list) -> Dict:
        importance = self.model.feature_importances_
        return dict(zip(feature_names, importance))

    def print_metrics(self) -> None:
        print("\n" + "=" * 60)
        print("Evaluation Metrics")
        print("=" * 60)
        print(f"Accuracy:  {self.metrics['accuracy']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall:    {self.metrics['recall']:.4f}")
        print(f"F1-Score:  {self.metrics['f1']:.4f}")
        print("\nConfusion Matrix:")
        print(self.metrics["confusion_matrix"])
        print("\nClassification Report:")
        print(self.metrics["classification_report"])
        print("=" * 60 + "\n")

    def print_feature_importance(self, feature_names: list) -> None:
        importance_dict = self.get_feature_importance(feature_names)
        sorted_features = sorted(
            importance_dict.items(), key=lambda x: x[1], reverse=True
        )

        print("\n" + "=" * 60)
        print("Feature Importance")
        print("=" * 60)
        for i, (feature, importance) in enumerate(sorted_features, 1):
            bar_length = int(importance * 50)
            bar = "█" * bar_length
            print(f"{i}. {feature:25s} {importance:6.4f} {bar}")
        print("=" * 60 + "\n")


class HybridClassifier:
    def __init__(self):
        self.rule_classifier = RuleBasedClassifier()
        self.ml_classifier = MLClassifier()

    def classify_hybrid(
        self,
        age_months: float,
        weight_kg: float,
        height_cm: float,
        haz: float,
        waz: float,
        whz: float,
        energy_pct: float,
        protein_g_per_kg: float,
        ml_prediction: int = None,
    ) -> Dict:
        """
        Classify using both rule-based and ML approaches.

        Args:
            age_months: Child age in months
            weight_kg: Weight in kg
            height_cm: Height in cm
            haz: Height-for-age Z-score
            waz: Weight-for-age Z-score
            whz: Weight-for-height Z-score
            energy_pct: Energy intake as % of requirement
            protein_g_per_kg: Protein intake in g/kg body weight
            ml_prediction: ML model prediction (0=low, 1=medium, 2=high)

        Returns:
            Dictionary with both classifications and recommendations
        """
        rule_risk, rule_explanation = self.rule_classifier.classify(
            age_months, weight_kg, height_cm, haz, waz, whz, energy_pct, protein_g_per_kg
        )

        ml_risk = RISK_LEVELS_REVERSE.get(ml_prediction, "low") if ml_prediction is not None else None

        return {
            "rule_based_risk": rule_risk,
            "ml_based_risk": ml_risk,
            "final_risk": rule_risk,  
            "rule_explanation": rule_explanation,
            "recommendation": self._get_recommendation(
                risk_level=rule_risk,
                age_months=age_months,
                weight_kg=weight_kg,
                energy_pct=energy_pct,
                protein_g_per_kg=protein_g_per_kg,
                whz=whz,
            ),
        }

    @staticmethod
    def _get_recommendation(
        risk_level: str,
        age_months: float,
        weight_kg: float,
        energy_pct: Optional[float] = None,
        protein_g_per_kg: Optional[float] = None,
        whz: Optional[float] = None,
    ) -> str:
        recommendations = []
        
        # Age-specific food suggestions
        if age_months < 6:
            age_food_tip = "Focus on exclusive breastfeeding or appropriate infant formula."
        elif age_months < 12:
            age_food_tip = "Continue breastfeeding with introduction of soft, mashed foods (porridge, mashed fruits/vegetables, egg yolk)."
        elif age_months < 24:
            age_food_tip = "Provide 3 main meals + 2 snacks daily. Include variety: rice/porridge, vegetables, protein (egg, fish, chicken, tofu), and fruits."
        else:
            age_food_tip = "Ensure balanced meals 3x daily with protein sources (egg, fish, meat, legumes), vegetables, fruits, and whole grains."

        # === LOW RISK ===
        if risk_level == "low":
            # Check for overweight/obesity concern first
            if whz is not None and whz >= 3.0:
                recommendations.append("OBESITY ALERT: Child is obese (WHZ ≥ 3.0). Immediate dietary changes needed:")
                recommendations.append("• Reduce fried foods, sugary drinks (juice boxes, soft drinks), and processed snacks")
                recommendations.append("• Replace with water, fresh fruits, and vegetables")
                recommendations.append("• Increase physical activity appropriate for age")
                recommendations.append("• Consult a pediatric nutritionist within 2 weeks")
                recommendations.append("• Monthly weight monitoring required")
                return " ".join(recommendations)
            
            if whz is not None and whz >= 2.0:
                recommendations.append("OVERWEIGHT WARNING: Child shows overweight status (WHZ ≥ 2.0).")
                recommendations.append("• Reduce calorie-dense foods: fried snacks, sweetened beverages, cookies/cakes")
                recommendations.append("• Prioritize balanced meals with more vegetables and lean protein")
                recommendations.append("• Encourage active play (30+ mins daily)")
                recommendations.append("• Monitor weight every 2 weeks")
                if energy_pct is not None and energy_pct >= 110:
                    recommendations.append(f"• Current energy intake ({energy_pct:.0f}%) is excessive - reduce portion sizes")
                return " ".join(recommendations)
            
            # Excess energy without overweight yet
            if energy_pct is not None:
                if energy_pct >= 130:
                    recommendations.append(f"HIGH CALORIE ALERT: Energy intake is {energy_pct:.0f}% of requirement - significantly above normal.")
                    recommendations.append("If continued, this may lead to childhood obesity. Actions needed:")
                    recommendations.append("• Reduce portion sizes by 20-30%")
                    recommendations.append("• Eliminate or limit: sugary drinks, fried foods, instant noodles, packaged snacks")
                    recommendations.append("• Increase vegetables in every meal")
                    recommendations.append("• Use smaller plates/bowls to control portions")
                    recommendations.append("• Monitor weight monthly")
                    return " ".join(recommendations)
                
                if energy_pct >= 120:
                    recommendations.append(f"Energy intake is moderately high ({energy_pct:.0f}% of requirement).")
                    recommendations.append("Consider these adjustments to prevent future weight issues:")
                    recommendations.append("• Reduce sugary snacks and drinks between meals")
                    recommendations.append("• Choose grilled/steamed foods over fried")
                    recommendations.append("• Ensure adequate vegetables (half the plate)")
                    recommendations.append("• Check-up in 2 months to monitor weight trend")
                    return " ".join(recommendations)
                
                if energy_pct >= 110:
                    recommendations.append(f"Energy intake is slightly elevated ({energy_pct:.0f}% of requirement).")
                    recommendations.append("This is acceptable if child is very active, but monitor:")
                    recommendations.append("• Watch for rapid weight gain over the next months")
                    recommendations.append("• Limit sugary treats to occasional consumption")
                    recommendations.append("• Routine check-up in 3 months")
                    return " ".join(recommendations)
            
            # Normal status
            recommendations.append("HEALTHY STATUS: Child shows normal nutritional indicators.")
            recommendations.append(age_food_tip)
            recommendations.append("Continue current feeding practices with:")
            recommendations.append("• Regular meal times (3 main meals + 2 healthy snacks)")
            recommendations.append("• Variety of food groups daily")
            recommendations.append("• Clean water as main beverage")
            recommendations.append("• Routine growth monitoring every 3 months")
            return " ".join(recommendations)

        # === MEDIUM RISK ===
        if risk_level == "medium":
            recommendations.append("ATTENTION NEEDED: Child shows borderline nutritional status.")
            
            # Check specific issues
            low_energy = energy_pct is not None and energy_pct < ENERGY_ADEQUATE
            low_protein = protein_g_per_kg is not None and protein_g_per_kg < PROTEIN_ADEQUATE
            
            if low_energy and low_protein:
                recommendations.append("Both energy and protein intake are below optimal levels.")
                recommendations.append("Dietary improvements needed:")
                recommendations.append("• Increase meal frequency: add 1-2 nutritious snacks daily")
                recommendations.append("• Add calorie-dense healthy foods: avocado, peanut butter, cheese")
                recommendations.append("• Include protein in every meal: eggs, fish, chicken, tofu, legumes")
                recommendations.append("• Consider fortified foods or supplements (consult doctor first)")
            elif low_energy:
                recommendations.append(f"Energy intake ({energy_pct:.0f}%) is below the recommended 90%.")
                recommendations.append("To increase caloric intake:")
                recommendations.append("• Add healthy fats: cooking oil, butter, coconut milk in meals")
                recommendations.append("• Offer nutrient-dense snacks: banana with peanut butter, cheese crackers")
                recommendations.append("• Don't skip meals - ensure 3 main meals + 2 snacks")
            elif low_protein:
                recommendations.append(f"Protein intake ({protein_g_per_kg:.2f} g/kg) is below optimal ({PROTEIN_ADEQUATE} g/kg).")
                recommendations.append("To increase protein intake:")
                recommendations.append("• Add protein to every meal: egg, fish, chicken, tempeh, tofu")
                recommendations.append("• Give milk or yogurt as snacks")
                recommendations.append("• Include legumes: red beans, green beans, lentils")
            else:
                recommendations.append("Z-scores indicate borderline growth measurements.")
                recommendations.append("Focus on overall diet quality:")
            
            recommendations.append(age_food_tip)
            recommendations.append("• Monthly weight and height monitoring recommended")
            recommendations.append("• Consider consultation with a nutritionist if no improvement in 1-2 months")
            return " ".join(recommendations)

        # === HIGH RISK ===
        if risk_level == "high":
            recommendations.append("HIGH RISK - URGENT ACTION REQUIRED")
            recommendations.append("Child shows significant signs of malnutrition requiring immediate attention.")
            recommendations.append("")
            recommendations.append("IMMEDIATE ACTIONS:")
            recommendations.append("• Consult a pediatrician or nutritionist within 1 week")
            recommendations.append("• If child is lethargic, has swelling, or refuses food - seek medical care TODAY")
            recommendations.append("")
            recommendations.append("DIETARY INTERVENTIONS:")
            recommendations.append("• Increase meal frequency to 5-6 small meals daily")
            recommendations.append("• Add high-calorie foods: full-fat milk, eggs, peanut butter, cheese, avocado")
            recommendations.append("• Include protein in every meal: egg, fish, chicken, tofu, legumes")
            recommendations.append("• Add oil/butter to foods for extra calories")
            recommendations.append("• Consider therapeutic foods (like F-100) if recommended by health worker")
            recommendations.append("")
            recommendations.append(age_food_tip)
            recommendations.append("")
            recommendations.append("MONITORING:")
            recommendations.append("• Weekly weight monitoring is essential")
            recommendations.append("• Watch for warning signs: persistent diarrhea, fever, loss of appetite, swelling")
            recommendations.append("• Keep a food diary to track daily intake")
            return " ".join(recommendations)

        return "Unable to generate recommendation. Please consult a healthcare professional."
