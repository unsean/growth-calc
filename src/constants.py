# WHO Z-Score Thresholds for Nutritional Status
WHO_THRESHOLDS = {
    "severe_malnutrition": -3.0,
    "moderate_malnutrition": -2.0,
    "normal": -2.0,
}

# Energy Requirements (% of recommended daily intake)
ENERGY_THRESHOLDS = {
    "adequate": 90,      # >= 90%
    "borderline": 70,    # 70-89%
    "deficient": 0,      # < 70%
}

# Protein Requirements (g/kg body weight)
PROTEIN_THRESHOLDS = {
    "adequate": 1.1,     # >= 1.1 g/kg
    "borderline": 0.9,   # 0.9-1.1 g/kg
    "deficient": 0,      # < 0.9 g/kg
}

# Risk Classification Mapping
RISK_LEVELS = {
    "high": 2,
    "medium": 1,
    "low": 0,
}

RISK_LEVELS_REVERSE = {v: k for k, v in RISK_LEVELS.items()}

# Age Groups for Nutritional Recommendations (in months)
AGE_GROUPS = {
    "0-6": (0, 6),
    "6-12": (6, 12),
    "12-24": (12, 24),
    "24-60": (24, 60),
}

# Daily Energy Requirements by Age (kcal) - WHO/FAO recommendations
ENERGY_REQUIREMENTS = {
    "0-6": 500,
    "6-12": 700,
    "12-24": 1000,
    "24-60": 1400,
}

# Daily Protein Requirements by Age (g) - WHO recommendations
PROTEIN_REQUIREMENTS = {
    "0-6": 9.1,
    "6-12": 13.5,
    "12-24": 13.5,
    "24-60": 20,
}
