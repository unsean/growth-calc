from flask import Flask, render_template, request, jsonify
import numpy as np
import logging
from pathlib import Path
from risk_classifier import RuleBasedClassifier, MLClassifier, HybridClassifier
from model_manager import ModelManager
from constants import ENERGY_REQUIREMENTS, PROTEIN_REQUIREMENTS, AGE_GROUPS
from zscore_calculator import calculate_all_zscores

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder=str(Path(__file__).parent.parent / "templates"))

# Initialize classifiers
rule_classifier = RuleBasedClassifier()
hybrid_classifier = HybridClassifier()

# Load pre-trained ML model
model_manager = ModelManager()
loaded_model = model_manager.load_model()
model_metadata = model_manager.load_metadata()

if loaded_model:
    ml_classifier = MLClassifier()
    ml_classifier.model = loaded_model
    print("ML model loaded from disk")
else:
    print("ML model not found. Train the model first with: python run.py")
    ml_classifier = None


def get_age_group(age_months: float) -> str:
    for group, (min_age, max_age) in AGE_GROUPS.items():
        if min_age <= age_months < max_age:
            return group
    return "24-60"


def get_energy_requirement(age_months: float) -> float:
    age_group = get_age_group(age_months)
    return ENERGY_REQUIREMENTS.get(age_group, 1400)


def get_protein_requirement(age_months: float) -> float:
    age_group = get_age_group(age_months)
    return PROTEIN_REQUIREMENTS.get(age_group, 20)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/assess", methods=["POST"])
def assess_risk():
    try:
        data = request.get_json()
        logger.info(f"Received assessment request: age={data.get('age_months')}, gender={data.get('gender')}")

        # Extract and validate input
        age_months = float(data.get("age_months", 0))
        gender = data.get("gender", "M").upper()
        weight_kg = float(data.get("weight_kg", 0))
        height_cm = float(data.get("height_cm", 0))
        energy_intake_kcal = float(data.get("energy_intake_kcal", 0))
        protein_intake_g = float(data.get("protein_intake_g", 0))
        
        logger.debug(f"Parsed input: age={age_months}, gender={gender}, weight={weight_kg}, height={height_cm}")

        # Validate ranges
        if not (0 <= age_months <= 60):
            return jsonify({"error": "Age must be between 0-60 months"}), 400
        if not (2 <= weight_kg <= 30):
            return jsonify({"error": "Weight must be between 2-30 kg"}), 400
        if not (40 <= height_cm <= 120):
            return jsonify({"error": "Height must be between 40-120 cm"}), 400

        # Calculate derived metrics
        bmi = weight_kg / ((height_cm / 100) ** 2)

        # Get requirements
        energy_req = get_energy_requirement(age_months)
        protein_req = get_protein_requirement(age_months)

        # Calculate percentages
        energy_pct = (energy_intake_kcal / energy_req * 100) if energy_req > 0 else 0
        protein_g_per_kg = protein_intake_g / weight_kg if weight_kg > 0 else 0

        # Calculate Z-scores using WHO reference data (gender-aware)
        haz, waz, whz = calculate_all_zscores(
            age_months=age_months,
            weight_kg=weight_kg,
            height_cm=height_cm,
            gender=gender
        )
        
        logger.debug(f"Z-scores calculated: HAZ={haz:.2f}, WAZ={waz:.2f}, WHZ={whz:.2f}")

        # Get classification
        risk, explanation = rule_classifier.classify(
            age_months=age_months,
            weight_kg=weight_kg,
            height_cm=height_cm,
            haz=haz,
            waz=waz,
            whz=whz,
            energy_pct=energy_pct,
            protein_g_per_kg=protein_g_per_kg,
        )

        # Get ML prediction if model is available
        ml_pred = None
        if ml_classifier:
            X_sample = np.array([[age_months, weight_kg, height_cm, bmi, haz, waz, whz, energy_pct, protein_g_per_kg]])
            ml_pred = ml_classifier.predict(X_sample)[0]

        # Get hybrid recommendation
        hybrid_result = hybrid_classifier.classify_hybrid(
            age_months=age_months,
            weight_kg=weight_kg,
            height_cm=height_cm,
            haz=haz,
            waz=waz,
            whz=whz,
            energy_pct=energy_pct,
            protein_g_per_kg=protein_g_per_kg,
            ml_prediction=ml_pred,
        )

        # Prepare response
        response = {
            "status": "success",
            "assessment": {
                "risk_level": risk.upper(),
                "color": get_risk_color(risk),
                "confidence": get_confidence_score(explanation),
                "explanation": {
                    "severe_indicators": explanation["severe_indicators"],
                    "moderate_indicators": explanation["moderate_indicators"],
                    "nutrition_indicators": explanation["nutrition_indicators"],
                },
            },
            "metrics": {
                "gender": gender,
                "age_months": age_months,
                "weight_kg": round(weight_kg, 2),
                "height_cm": round(height_cm, 2),
                "bmi": round(bmi, 2),
                "haz": round(haz, 2),
                "waz": round(waz, 2),
                "whz": round(whz, 2),
                "energy_pct": round(energy_pct, 1),
                "protein_g_per_kg": round(protein_g_per_kg, 2),
                "energy_intake_kcal": energy_intake_kcal,
                "protein_intake_g": protein_intake_g,
            },
            "requirements": {
                "daily_energy_kcal": energy_req,
                "daily_protein_g": protein_req,
            },
            "explanation": {
                "severe_indicators": explanation["severe_indicators"],
                "moderate_indicators": explanation["moderate_indicators"],
                "nutrition_indicators": explanation["nutrition_indicators"],
            },
            "recommendation": hybrid_result["recommendation"],
            "interpretation": get_risk_interpretation(risk),
        }

        logger.info(f"Assessment complete: risk={risk.upper()}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Assessment error: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 400


@app.route("/batch", methods=["POST"])
def batch_assess():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({"success": False, "error": "File must be CSV format"}), 400
        
        import io
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        import csv
        reader = csv.DictReader(stream)
        
        logger.info(f"Batch processing: CSV headers={reader.fieldnames}")
        predictions = []
        for row in reader:
            logger.debug(f"Processing row: {row}")
            try:
                age_months = float(row.get('Age_months', 0))
                gender = row.get('Gender', 'M').upper()
                weight_kg = float(row.get('Weight_kg', 0))
                height_cm = float(row.get('Height_cm', 0))
                
                # Use defaults for energy and protein, allow CSV override when available
                raw_energy = row.get('Energy_kcal')
                raw_protein = row.get('Protein_g')

                try:
                    energy_intake_kcal = float(raw_energy) if raw_energy not in (None, "") else 0.0
                except ValueError:
                    energy_intake_kcal = 0.0

                try:
                    protein_intake_g = float(raw_protein) if raw_protein not in (None, "") else 0.0
                except ValueError:
                    protein_intake_g = 0.0

                if energy_intake_kcal <= 0:
                    energy_intake_kcal = 900
                if protein_intake_g <= 0:
                    protein_intake_g = 15
                
                if not (0 <= age_months <= 60):
                    predictions.append({"error": f"Invalid age: {age_months}"})
                    continue
                if not (2 <= weight_kg <= 30):
                    predictions.append({"error": f"Invalid weight: {weight_kg}"})
                    continue
                if not (40 <= height_cm <= 120):
                    predictions.append({"error": f"Invalid height: {height_cm}"})
                    continue
                
                # Calculate metrics
                bmi = weight_kg / ((height_cm / 100) ** 2)
                energy_req = get_energy_requirement(age_months)
                # protein_req = get_protein_requirement(age_months)
                energy_pct = (energy_intake_kcal / energy_req * 100) if energy_req > 0 else 0
                protein_g_per_kg = protein_intake_g / weight_kg if weight_kg > 0 else 0
                
                haz, waz, whz = calculate_all_zscores(
                    age_months=age_months,
                    weight_kg=weight_kg,
                    height_cm=height_cm,
                    gender=gender
                )
                
                # Classify
                risk, explanation = rule_classifier.classify(
                    age_months, weight_kg, height_cm, haz, waz, whz, energy_pct, protein_g_per_kg
                )
                
                # Get hybrid recommendation
                hybrid_result = hybrid_classifier.classify_hybrid(
                    age_months=age_months,
                    weight_kg=weight_kg,
                    height_cm=height_cm,
                    haz=haz,
                    waz=waz,
                    whz=whz,
                    energy_pct=energy_pct,
                    protein_g_per_kg=protein_g_per_kg,
                    ml_prediction=None,
                )
                
                predictions.append({
                    "age_months": age_months,
                    "gender": gender,
                    "weight_kg": weight_kg,
                    "height_cm": height_cm,
                    "energy_intake_kcal": energy_intake_kcal,
                    "protein_intake_g": protein_intake_g,
                    "risk_level": risk.upper(),
                    "confidence": get_confidence_score(explanation),
                    "bmi": round(bmi, 2),
                    "haz": round(haz, 2),
                    "waz": round(waz, 2),
                    "whz": round(whz, 2),
                    "energy_pct": round(energy_pct, 1),
                    "protein_g_per_kg": round(protein_g_per_kg, 2),
                    "recommendation": hybrid_result["recommendation"],
                    "interpretation": get_risk_interpretation(risk),
                    "explanation": {
                        "severe_indicators": explanation["severe_indicators"],
                        "moderate_indicators": explanation["moderate_indicators"],
                        "nutrition_indicators": explanation["nutrition_indicators"],
                    }
                })
            except Exception as e:
                predictions.append({"error": str(e)})
        
        return jsonify({"success": True, "count": len(predictions), "predictions": predictions})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/guidelines", methods=["GET"])
def get_guidelines():
    guidelines = {}
    for group, (min_age, max_age) in AGE_GROUPS.items():
        guidelines[group] = {
            "age_range": f"{min_age}-{max_age} months",
            "energy_kcal": ENERGY_REQUIREMENTS[group],
            "protein_g": PROTEIN_REQUIREMENTS[group],
        }
    return jsonify(guidelines)


@app.route("/info", methods=["GET"])
def get_model_info():
    if model_metadata:
        return jsonify({
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "training_samples": model_metadata.get("training_samples"),
            "testing_samples": model_metadata.get("testing_samples"),
            "features": len(model_metadata.get("features", [])),
            "classes": model_metadata.get("classes", ["low", "medium", "high"])
        })
    return jsonify({
        "error": "Model metadata not available",
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "training_samples": None,
        "testing_samples": None,
        "features": 0,
        "classes": ["low", "medium", "high"]
    })


@app.route("/api/batch", methods=["POST"])
def api_batch_assess():
    return batch_assess()


@app.route("/api/info", methods=["GET"])
def api_get_model_info():
    return get_model_info()


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "service": "malnutrition-detection-api"})


def get_risk_color(risk_level: str) -> str:
    colors = {
        "low": "#10b981",  # Green
        "medium": "#f59e0b",  # Amber
        "high": "#ef4444",  # Red
    }
    return colors.get(risk_level.lower(), "#6b7280")


def get_confidence_score(explanation: dict) -> float:
    indicator_count = (
        len(explanation["severe_indicators"])
        + len(explanation["moderate_indicators"])
        + len(explanation["nutrition_indicators"])
    )
    # Confidence based on number of supporting indicators
    # 0 indicators (LOW RISK) = 95% confidence (very clear)
    # 1 indicator = 70% confidence
    # 2+ indicators = 90% confidence
    if indicator_count == 0:
        confidence = 95.0  # Clear LOW RISK
    elif indicator_count == 1:
        confidence = 70.0  # Some concern
    else:
        confidence = min(50 + indicator_count * 15, 100)  # Multiple indicators
    return round(confidence, 1)


def get_risk_interpretation(risk_level: str) -> str:
    interpretations = {
        "low": (
            "Child's growth indicators (height-for-age, weight-for-age, weight-for-height) are within normal WHO standards. "
            "Nutritional intake appears adequate. This indicates healthy growth and development. "
            "Continue current feeding practices, ensure dietary variety, and maintain routine growth monitoring every 3 months. "
            "If any concerns arise (appetite changes, frequent illness, growth slowdown), consult a healthcare provider."
        ),
        "medium": (
            "Child shows one or more borderline indicators that require attention. This could include: "
            "moderate stunting (short for age), moderate underweight, borderline energy/protein intake, or early signs of overweight. "
            "While not immediately critical, this status can worsen without intervention. "
            "Recommended actions: review and improve diet quality, increase monitoring to monthly, and consider professional nutritional guidance. "
            "With proper dietary adjustments, most children improve to normal status within 2-3 months."
        ),
        "high": (
            "Child shows significant nutritional concerns requiring urgent attention. This may include: "
            "severe stunting (HAZ < -3), severe underweight (WAZ < -3), severe wasting (WHZ < -3), "
            "or a combination of moderate indicators with nutritional deficiencies. "
            "High-risk status is associated with increased vulnerability to infections, developmental delays, and serious health complications. "
            "Immediate actions needed: consult a pediatrician or nutritionist within 1 week, implement intensive dietary interventions, "
            "and monitor weight weekly. Early intervention is critical for recovery and preventing long-term consequences."
        ),
    }
    return interpretations.get(risk_level.lower(), "Unable to interpret risk level. Please consult a healthcare professional.")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
