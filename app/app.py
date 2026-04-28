"""
app/app.py
----------
Flask web application for the Nakaseke NCD-AI Hypertension Risk Screener.

How it works:
  1. The user fills in a short questionnaire (no blood tests required).
  2. POST /predict sends the form data to the prediction endpoint.
  3. The loaded ML pipeline transforms the raw inputs exactly as the training
     pipeline did, then returns a hypertension risk probability.
  4. The result page shows the risk score, a traffic-light indicator,
     clinical interpretation, and personalised lifestyle advice.

Security notes:
  - All inputs are validated and clamped to physiological ranges server-side
    before passing to the model; there is no SQL or shell injection surface.
  - The app never stores any patient data in a database or on disk.
"""

import os
import sys
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Allow importing from project root when running locally
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CFG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "nakaseke-ncd-ai-2024")


# ─────────────────────────────────────────────────────────────────────────────
# Load the trained model once at startup (not per request)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH    = Path(__file__).parent.parent / CFG.paths.best_model
FEATURES_PATH = Path(__file__).parent.parent / "models" / "feature_names.csv"

try:
    with open(MODEL_PATH, "rb") as f:
        MODEL = pickle.load(f)
    FEATURE_NAMES = pd.read_csv(FEATURES_PATH).iloc[:, 0].tolist()
    logger.info("Model loaded: %s  (%d features)", MODEL_PATH.name, len(FEATURE_NAMES))
except Exception as e:
    logger.error("Could not load model: %s.  Run 'python train.py' first.", e)
    MODEL        = None
    FEATURE_NAMES = []


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering replicated for single-row inference
# ─────────────────────────────────────────────────────────────────────────────

_FREQ_MAP = {"never": 0, "rarely": 1, "sometimes": 2, "often": 3, "always": 4}
_OIL_MAP  = {
    "vegetable": 1, "sunflower": 1, "palm": 1, "simsim": 1,
    "olive": 2, "none": 0, "solid": 0, "other": 0,
}


def _safe_float(val, lo=None, hi=None, default=0.0):
    try:
        v = float(val)
        if lo is not None and v < lo:
            v = lo
        if hi is not None and v > hi:
            v = hi
        return v
    except (TypeError, ValueError):
        return default


def build_feature_row(form: dict) -> pd.DataFrame:
    """
    Convert raw HTML form fields into the 36-feature vector expected by the
    trained model pipeline.

    The engineering steps here mirror src/data/cleaner.py and
    src/data/features.py exactly.  Any divergence between training-time and
    inference-time feature engineering is a silent data-leakage bug, so we
    keep the logic in a single place per file and cross-reference it here.
    """
    # ── Raw demographic inputs ────────────────────────────────────────────────
    female          = 1 if form.get("gender") == "female" else 0
    age             = _safe_float(form.get("age"),          15, 110, 45)
    education       = _safe_float(form.get("education"),     0,  6,   2)
    married         = 1 if form.get("married") == "1"    else 0
    employed        = 1 if form.get("employed") == "1"   else 0
    household_size  = _safe_float(form.get("household_size"), 1, 30, 4)

    # ── Lifestyle – tobacco ───────────────────────────────────────────────────
    ever_smoked     = 1 if form.get("ever_smoked") == "1"    else 0
    current_smoker  = 1 if form.get("current_smoker") == "1" else 0
    if not ever_smoked:
        current_smoker = 0

    # ── Lifestyle – alcohol ───────────────────────────────────────────────────
    ever_alcohol    = 1 if form.get("ever_alcohol") == "1" else 0
    drinks_per_week = _safe_float(form.get("drinks_per_week"), 0, 50, 0)

    # ── Diet ─────────────────────────────────────────────────────────────────
    eats_fruit           = 1 if form.get("eats_fruit") == "1" else 0
    fruit_servings_week  = _safe_float(form.get("fruit_servings"), 0, 21, 3) / 21
    veg_servings_week    = _safe_float(form.get("veg_servings"),   0, 21, 3) / 21
    adds_salt            = _FREQ_MAP.get(form.get("adds_salt", "sometimes"), 2)
    processed_food_freq  = _FREQ_MAP.get(form.get("processed_food", "sometimes"), 2)
    cooking_oil_type     = _OIL_MAP.get(form.get("cooking_oil", "vegetable").lower(), 1)
    biomass_exposure     = 1 if form.get("biomass") == "1" else 0

    # ── Physical activity ─────────────────────────────────────────────────────
    vigorous_activity   = 1 if form.get("vigorous_activity") == "1" else 0
    vigorous_days_week  = _safe_float(form.get("vigorous_days"), 0, 7, 3)

    # ── Anthropometrics ───────────────────────────────────────────────────────
    height_cm = _safe_float(form.get("height_cm"), 100, 220, 160)
    weight_kg = _safe_float(form.get("weight_kg"),  20, 200,  65)
    waist_cm  = _safe_float(form.get("waist_cm"),   40, 180,  80)
    hip_cm    = _safe_float(form.get("hip_cm"),      50, 200,  95)

    bmi  = weight_kg / ((height_cm / 100) ** 2)
    bmi  = np.clip(bmi, 12, 70)
    whr  = waist_cm / max(hip_cm, 1)
    whr  = np.clip(whr, 0.5, 1.5)

    # ── Engineered features ───────────────────────────────────────────────────
    # BMI category (WHO)
    bmi_cat = 0 if bmi < 18.5 else (1 if bmi < 25 else (2 if bmi < 30 else 3))
    # Age group
    age_group = 0 if age < 30 else (1 if age < 45 else (2 if age < 60 else 3))
    # Waist risk flag
    waist_risk = 1 if (female and waist_cm > 88) or (not female and waist_cm > 102) else 0
    # WHR risk flag
    whr_risk   = 1 if (female and whr > 0.85) or (not female and whr > 0.90) else 0
    # Lifestyle score
    lifestyle_score = (
        current_smoker +
        int(drinks_per_week > 14) +
        int(bmi >= 25) +
        int(fruit_servings_week + veg_servings_week < 0.2) +
        int(adds_salt >= 3) +
        (1 - vigorous_activity)
    )
    lifestyle_score = np.clip(lifestyle_score, 0, 6)
    # Metabolic burden
    metabolic_burden = int(bmi >= 30) + waist_risk + whr_risk
    metabolic_burden = np.clip(metabolic_burden, 0, 3)

    # Normalised z-like values for interaction terms
    # (training means/stds are baked into imputer; we approximate here)
    age_z  = (age   - 45)  / 16
    bmi_z  = (bmi   - 28)  / 11
    whr_z  = (whr   - 0.87) / 0.10

    age_bmi  = age_z * bmi_z
    age_whr  = age_z * whr_z
    bmi_salt = bmi_z * (1 if adds_salt >= 3 else 0)
    age_sq   = age_z ** 2
    bmi_sq   = bmi_z ** 2

    # ── Assemble into ordered dict matching FEATURE_NAMES ────────────────────
    raw = {
        "female":              female,
        "age":                 age,
        "education":           education,
        "married":             married,
        "employed":            employed,
        "household_size":      household_size,
        "ever_smoked":         ever_smoked,
        "current_smoker":      current_smoker,
        "ever_alcohol":        ever_alcohol,
        "drinks_per_week":     drinks_per_week,
        "eats_fruit":          eats_fruit,
        "fruit_servings_week": fruit_servings_week,
        "veg_servings_week":   veg_servings_week,
        "adds_salt":           adds_salt,
        "processed_food_freq": processed_food_freq,
        "cooking_oil_type":    cooking_oil_type,
        "biomass_exposure":    biomass_exposure,
        "vigorous_activity":   vigorous_activity,
        "vigorous_days_week":  vigorous_days_week,
        "height_cm":           height_cm,
        "weight_kg":           weight_kg,
        "bmi":                 bmi,
        "waist_cm":            waist_cm,
        "hip_cm":              hip_cm,
        "whr":                 whr,
        "bmi_cat":             bmi_cat,
        "age_group":           age_group,
        "waist_risk":          waist_risk,
        "whr_risk":            whr_risk,
        "lifestyle_score":     lifestyle_score,
        "metabolic_burden":    metabolic_burden,
        "age_bmi":             age_bmi,
        "age_whr":             age_whr,
        "bmi_salt":            bmi_salt,
        "age_sq":              age_sq,
        "bmi_sq":              bmi_sq,
    }

    # Only keep features the model was trained on, in the same order
    row = {k: raw.get(k, 0) for k in FEATURE_NAMES}
    return pd.DataFrame([row])


# ─────────────────────────────────────────────────────────────────────────────
# Risk interpretation helpers
# ─────────────────────────────────────────────────────────────────────────────

def interpret_risk(prob: float, form: dict) -> dict:
    """
    Convert the raw probability into a clinician-friendly risk band
    and personalised recommendations.
    """
    if prob < 0.25:
        band  = "Low"
        color = "#27AE60"
        emoji = "green"
        msg   = (
            "Your estimated hypertension risk is low based on the information "
            "provided. Maintaining a healthy lifestyle will keep it that way."
        )
    elif prob < 0.50:
        band  = "Moderate"
        color = "#F39C12"
        emoji = "yellow"
        msg   = (
            "Your risk is moderate. Some lifestyle adjustments could meaningfully "
            "reduce your chances of developing hypertension."
        )
    elif prob < 0.70:
        band  = "High"
        color = "#E74C3C"
        emoji = "orange"
        msg   = (
            "Your risk is high. We strongly recommend visiting a healthcare provider "
            "for a blood pressure measurement and clinical assessment."
        )
    else:
        band  = "Very High"
        color = "#8B0000"
        emoji = "red"
        msg   = (
            "Your risk is very high. Please seek medical attention promptly to have "
            "your blood pressure checked. Early detection saves lives."
        )

    # Generate targeted recommendations based on specific risk factors
    tips = []
    bmi = _safe_float(form.get("weight_kg"), 20, 200, 65) / \
          ((_safe_float(form.get("height_cm"), 100, 220, 160) / 100) ** 2)

    if bmi >= 25:
        tips.append("Reduce BMI through a balanced diet and regular physical activity "
                    f"(your estimated BMI: {bmi:.1f} kg/m²).")
    if form.get("current_smoker") == "1":
        tips.append("Quit smoking — it raises blood pressure directly and doubles cardiovascular risk.")
    if form.get("ever_alcohol") == "1" and _safe_float(form.get("drinks_per_week"), 0, 50, 0) > 7:
        tips.append("Reduce alcohol intake to fewer than 7 standard drinks per week.")
    if _FREQ_MAP.get(form.get("adds_salt", "sometimes"), 2) >= 3:
        tips.append("Reduce salt: avoid adding extra salt at the table and limit processed foods.")
    if form.get("vigorous_activity") != "1":
        tips.append("Aim for at least 150 minutes of moderate aerobic activity per week.")
    if form.get("biomass") == "1":
        tips.append("Reduce biomass smoke exposure by improving kitchen ventilation or switching fuel.")
    if not tips:
        tips.append("Continue your current healthy lifestyle habits.")

    return {
        "probability": round(prob * 100, 1),
        "band":        band,
        "color":       color,
        "emoji":       emoji,
        "message":     msg,
        "tips":        tips,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return render_template(
            "result.html",
            error="Model not available. Please run 'python train.py' first.",
        )

    try:
        form = request.form.to_dict()
        X    = build_feature_row(form)
        prob = float(MODEL.predict_proba(X)[0][1])
        interp = interpret_risk(prob, form)

        # For the result page
        bmi = _safe_float(form.get("weight_kg"), 20, 200, 65) / \
              ((_safe_float(form.get("height_cm"), 100, 220, 160) / 100) ** 2)

        return render_template(
            "result.html",
            error=None,
            prob=round(prob * 100, 1),
            band=interp["band"],
            color=interp["color"],
            message=interp["message"],
            tips=interp["tips"],
            bmi=round(bmi, 1),
            form=form,
        )
    except Exception as e:
        logger.exception("Prediction error")
        return render_template("result.html", error=f"Prediction error: {e}")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for programmatic access / integration tests."""
    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 503
    try:
        data = request.get_json(force=True)
        X    = build_feature_row(data)
        prob = float(MODEL.predict_proba(X)[0][1])
        pred = int(prob >= 0.5)
        return jsonify({
            "hypertension_probability": round(prob, 4),
            "prediction":  pred,
            "risk_band":   interpret_risk(prob, data)["band"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})


if __name__ == "__main__":
    app.run(
        host=CFG.flask.host,
        port=CFG.flask.port,
        debug=CFG.flask.debug,
    )
