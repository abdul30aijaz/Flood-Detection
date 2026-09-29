import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Flood Risk Assessment", layout="centered")

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
MODEL_PATH = MODELS_DIR / "xgb_flood_model.joblib"
KMEANS_PATH = MODELS_DIR / "kmeans_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"

# Bounds and medians are from the raw training data, recovered by inverse-transforming
# the processed CSV with the saved training scaler.
INPUT_RANGES = {
    "rainfall": {"min": 0.59, "max": 418.41, "default": 68.745, "step": 0.1, "format": "%.2f"},
    "soil_moisture": {"min": 0.4, "max": 98.71, "default": 49.11, "step": 0.1, "format": "%.2f"},
    "humidity": {"min": 10.0, "max": 100.0, "default": 65.445, "step": 0.1, "format": "%.2f"},
    "ndvi": {"min": -0.3751, "max": 1.0, "default": 0.39395, "step": 0.001, "format": "%.4f"},
    "ndwi": {"min": -0.7751, "max": 0.6073, "default": -0.0763, "step": 0.001, "format": "%.4f"},
    "elevation": {"min": 5.78, "max": 1403.1, "default": 259.61, "step": 0.1, "format": "%.2f"},
    "slope": {"min": 0.03, "max": 36.08, "default": 4.65, "step": 0.01, "format": "%.2f"},
    "temperature": {"min": 10.82, "max": 45.0, "default": 26.96, "step": 0.1, "format": "%.2f"},
}

LOW_RISK_THRESHOLD = 0.33
HIGH_RISK_THRESHOLD = 0.66
CLUSTER_RISK_LABELS = {0: "High", 1: "Medium", 2: "Low"}


@st.cache_resource
def load_artifacts():
    required_paths = [MODEL_PATH, KMEANS_PATH, SCALER_PATH, FEATURE_COLUMNS_PATH]
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        missing_files = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in missing_paths)
        raise FileNotFoundError(f"Required model file(s) are missing: {missing_files}")

    with FEATURE_COLUMNS_PATH.open("r", encoding="utf-8") as feature_file:
        feature_columns = json.load(feature_file)
    if not isinstance(feature_columns, list) or "cluster_label" not in feature_columns:
        raise ValueError("feature_columns.json must be a list containing 'cluster_label'.")

    environmental_features = [
        feature for feature in feature_columns if feature != "cluster_label"
    ]
    if set(environmental_features) != set(INPUT_RANGES):
        raise ValueError("feature_columns.json does not match the app's eight input fields.")

    return (
        joblib.load(MODEL_PATH),
        joblib.load(KMEANS_PATH),
        joblib.load(SCALER_PATH),
        feature_columns,
        environmental_features,
    )


st.title("Flood Risk Assessment")
st.write("Enter environmental measurements to estimate flood probability.")

try:
    artifacts = load_artifacts()
except FileNotFoundError as error:
    st.error(f"Model files are missing. Add the Stage 3 model artifacts to the models folder. {error}")
    artifacts = None
except Exception as error:
    st.error(
        f"Could not load model artifacts: {error}. "
        "Check that the XGBoost version matches the version used to save the classifier."
    )
    artifacts = None

with st.form("flood_risk_form"):
    submitted_values = {}
    left_column, right_column = st.columns(2)
    for index, (feature, limits) in enumerate(INPUT_RANGES.items()):
        column = left_column if index % 2 == 0 else right_column
        with column:
            submitted_values[feature] = st.number_input(
                feature.replace("_", " ").title(),
                min_value=float(limits["min"]),
                max_value=float(limits["max"]),
                value=float(limits["default"]),
                step=float(limits["step"]),
                format=limits["format"],
            )
    submitted = st.form_submit_button("Assess flood risk", type="primary", use_container_width=True)

if submitted and artifacts is not None:
    xgb_model, kmeans_model, scaler, feature_columns, environmental_features = artifacts
    try:
        input_frame = pd.DataFrame(
            [[submitted_values[feature] for feature in environmental_features]],
            columns=environmental_features,
        )
        scaled_input = scaler.transform(input_frame)
        cluster_label = int(kmeans_model.predict(scaled_input)[0])

        model_input = pd.DataFrame(scaled_input, columns=environmental_features)
        model_input["cluster_label"] = cluster_label
        model_input = model_input.reindex(columns=feature_columns)

        probability_index = list(xgb_model.classes_).index(1)
        flood_probability = float(xgb_model.predict_proba(model_input)[0, probability_index])
        if flood_probability < LOW_RISK_THRESHOLD:
            risk_level, badge_color = "Low", "#247a45"
        elif flood_probability < HIGH_RISK_THRESHOLD:
            risk_level, badge_color = "Medium", "#9a6800"
        else:
            risk_level, badge_color = "High", "#b42318"

        cluster_risk = CLUSTER_RISK_LABELS.get(cluster_label, "Unmapped")
        st.markdown(
            f'<span style="display:inline-block;padding:0.35rem 0.7rem;'
            f'background:{badge_color};color:#fff;border-radius:4px;font-weight:700">'
            f'{risk_level} risk</span>',
            unsafe_allow_html=True,
        )
        st.metric("Flood probability", f"{flood_probability:.4f} ({flood_probability:.2%})")
        st.write(
            f"Based on your inputs, this matches a {cluster_risk.lower()} risk cluster region."
        )
    except Exception as error:
        st.error(f"Could not calculate flood risk: {error}")