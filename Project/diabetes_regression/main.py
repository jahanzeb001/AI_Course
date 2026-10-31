import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# 1. PAGE CONFIGURATION & THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="Diabetes Progression Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .prediction-card {
        background: #ffffff;
        border: 2px solid #3b82f6;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .risk-badge {
        display: inline-block;
        padding: 6px 16px;
        font-size: 0.95rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-top: 10px;
    }
    .risk-low { background-color: #dcfce7; color: #166534; border: 1px solid #86efac; }
    .risk-mod { background-color: #fef9c3; color: #854d0e; border: 1px solid #fde047; }
    .risk-high { background-color: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. CONSTANTS & DATASET STATISTICS
# ==============================================================================
# Raw stats from the original diabetes dataset (Efron et al.)
# Used for seamless bidirectional conversion between clinical units & normalized values.
RAW_STATS = {
    "age": {"mean": 48.5181, "scale": 275.2896, "min": 19.0, "max": 79.0, "median": 50.0, "unit": "years"},
    "sex": {"mean": 1.4683, "scale": 10.4908, "min": 1.0, "max": 2.0, "median": 1.0, "unit": "1=F, 2=M"},
    "bmi": {"mean": 26.3758, "scale": 92.7806, "min": 18.0, "max": 42.2, "median": 25.7, "unit": "kg/m²"},
    "bp":  {"mean": 94.6470, "scale": 290.4570, "min": 62.0, "max": 133.0, "median": 93.0, "unit": "mm Hg (MAP)"},
    "s1":  {"mean": 189.1403, "scale": 726.7691, "min": 97.0, "max": 301.0, "median": 186.0, "unit": "mg/dL (Total Chol)"},
    "s2":  {"mean": 115.4391, "scale": 638.6747, "min": 41.6, "max": 242.4, "median": 113.0, "unit": "mg/dL (LDL)"},
    "s3":  {"mean": 49.7885, "scale": 271.6182, "min": 22.0, "max": 99.0, "median": 48.0, "unit": "mg/dL (HDL)"},
    "s4":  {"mean": 4.0702, "scale": 27.0994, "min": 2.0, "max": 9.09, "median": 4.0, "unit": "ratio (TCH)"},
    "s5":  {"mean": 4.6414, "scale": 10.9702, "min": 3.258, "max": 6.107, "median": 4.62, "unit": "log (Triglycerides)"},
    "s6":  {"mean": 91.2602, "scale": 241.4230, "min": 58.0, "max": 124.0, "median": 91.0, "unit": "mg/dL (Glucose)"},
}

FEATURE_COLS = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

FEATURE_LABELS = {
    "age": "Age (years)",
    "sex": "Sex",
    "bmi": "Body Mass Index (BMI)",
    "bp": "Mean Arterial Blood Pressure (BP)",
    "s1": "Total Serum Cholesterol (s1 / tc)",
    "s2": "Low-Density Lipoproteins (s2 / ldl)",
    "s3": "High-Density Lipoproteins (s3 / hdl)",
    "s4": "Total Cholesterol / HDL (s4 / tch)",
    "s5": "Log Serum Triglycerides (s5 / ltg)",
    "s6": "Blood Glucose Level (s6 / glu)",
}

TARGET_STATS = {
    "min": 25.0,
    "max": 346.0,
    "mean": 152.13,
    "std": 77.09,
    "q25": 87.0,
    "q50": 140.5,
    "q75": 211.5,
}

MODEL_PATH = "best_diabetes_model.joblib"

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================
@st.cache_resource
def load_trained_model():
    """Load the serialized scikit-learn pipeline."""
    # Look in the script directory or current working directory
    possible_paths = [
        MODEL_PATH,
        os.path.join(os.path.dirname(__file__), MODEL_PATH),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            try:
                model = joblib.load(p)
                return model, p
            except Exception as e:
                st.error(f"Error loading model from {p}: {e}")
                return None, None
    return None, None


def clinical_to_normalized(raw_dict):
    """Convert real-world clinical values into scikit-learn diabetes scaled values."""
    scaled = {}
    for col, val in raw_dict.items():
        if col in RAW_STATS:
            scaled[col] = (val - RAW_STATS[col]["mean"]) / RAW_STATS[col]["scale"]
    return scaled


def normalized_to_clinical(scaled_dict):
    """Convert scikit-learn diabetes scaled values back to real-world clinical units."""
    raw = {}
    for col, val in scaled_dict.items():
        if col in RAW_STATS:
            raw[col] = val * RAW_STATS[col]["scale"] + RAW_STATS[col]["mean"]
    return raw


def get_risk_assessment(score):
    """Categorize disease progression risk index."""
    if score < 100:
        return (
            "Mild Progression / Low Risk",
            "risk-low",
            "Disease progression is projected to be mild/slow over the next year. Maintain healthy dietary habits and routine annual check-ups.",
        )
    elif score <= 180:
        return (
            "Moderate Progression / Moderate Risk",
            "risk-mod",
            "Disease progression is at a moderate rate. Recommend glycemic control, active lipid management, and routine 6-month clinical follow-up.",
        )
    else:
        return (
            "Accelerated Progression / High Risk",
            "risk-high",
            "Disease progression is markedly elevated. Clinical evaluation for intensive therapeutic intervention and lifestyle modification is strongly advised.",
        )


def compute_feature_contributions(pipeline, normalized_df):
    """Compute linear contributions (beta_i * scaled_x_i) from the Ridge pipeline."""
    try:
        scaler = pipeline.named_steps["scaler"]
        ridge = pipeline.named_steps["model"]

        # Transform inputs through StandardScaler
        X_scaled = scaler.transform(normalized_df)
        coefs = ridge.coef_
        intercept = ridge.intercept_

        contributions = X_scaled[0] * coefs
        contrib_df = pd.DataFrame({
            "Feature": [FEATURE_LABELS[c] for c in FEATURE_COLS],
            "Code": FEATURE_COLS,
            "Contribution": contributions,
            "Coefficient": coefs,
            "Standardized_Input": X_scaled[0],
        }).sort_values(by="Contribution", ascending=False)
        return contrib_df, intercept
    except Exception:
        return None, None


# ==============================================================================
# 4. APPLICATION LAYOUT & HEADER
# ==============================================================================
st.markdown('<div class="main-header">🩺 Diabetes Disease Progression Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Clinical decision support & 1-year disease progression forecasting powered by the optimized Ridge Regression pipeline from <code>diabetes_regression_project.ipynb</code>.</div>',
    unsafe_allow_html=True,
)

# Load Model
model, loaded_path = load_trained_model()

if model is None:
    st.error(
        f"⚠️ Could not locate `{MODEL_PATH}`. Please ensure the model file is placed in the project directory."
    )
    st.stop()

# Header metrics row
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
with col_m1:
    st.metric("Best Model", "Ridge Regression", "Tuned with LSQR")
with col_m2:
    st.metric("Cross-Val R²", "0.4808", "5-Fold CV")
with col_m3:
    st.metric("Test R²", "0.4541", "Hold-out split (20%)")
with col_m4:
    st.metric("Test RMSE", "53.78", "Progression units")
with col_m5:
    st.metric("Test MAE", "42.81", "progression units")

st.write("---")

# ==============================================================================
# 5. SIDEBAR: CONTROLS & PRESETS
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    input_mode = st.radio(
        "Input Mode:",
        ["Clinical Units (Recommended)", "Normalized Features (Sklearn format)"],
        help="Choose whether to enter measurements in familiar clinical units (e.g. years, kg/m², mg/dL) or raw zero-centered normalized values.",
    )

    st.write("---")
    st.subheader("📋 Preset Patient Profiles")
    st.caption("Quickly populate the input fields with representative patient profiles:")

    preset = st.selectbox(
        "Choose a Preset Profile:",
        [
            "Custom Input",
            "Healthy / Low-Progression Patient",
            "Average Cohort Patient",
            "Elevated Risk Patient",
            "Test Sample A (Actual Target: 219.0)",
            "Test Sample B (Actual Target: 70.0)",
        ],
    )

    st.write("---")
    st.subheader("📦 Model Architecture")
    st.code(
        "Pipeline(\n"
        "  StandardScaler(),\n"
        "  Ridge(alpha=1.0,\n"
        "        solver='lsqr')\n"
        ")",
        language="python",
    )
    st.caption(f"Loaded from: `{os.path.basename(loaded_path)}`")

# Preset value mapping in clinical units
presets_clinical = {
    "Healthy / Low-Progression Patient": {
        "age": 35.0, "sex": "Female", "bmi": 21.5, "bp": 80.0,
        "s1": 160.0, "s2": 90.0, "s3": 65.0, "s4": 2.5, "s5": 3.80, "s6": 78.0,
    },
    "Average Cohort Patient": {
        "age": 48.5, "sex": "Male", "bmi": 26.4, "bp": 94.6,
        "s1": 189.1, "s2": 115.4, "s3": 49.8, "s4": 4.07, "s5": 4.64, "s6": 91.3,
    },
    "Elevated Risk Patient": {
        "age": 62.0, "sex": "Male", "bmi": 34.0, "bp": 112.0,
        "s1": 240.0, "s2": 155.0, "s3": 32.0, "s4": 7.5, "s5": 5.45, "s6": 118.0,
    },
    "Test Sample A (Actual Target: 219.0)": {
        # Raw converted from X_test sample 0
        "age": 59.0, "sex": "Male", "bmi": 32.1, "bp": 101.0,
        "s1": 157.0, "s2": 93.2, "s3": 38.0, "s4": 4.0, "s5": 4.86, "s6": 87.0,
    },
    "Test Sample B (Actual Target: 70.0)": {
        # Raw converted from X_test sample 1
        "age": 48.0, "sex": "Female", "bmi": 21.6, "bp": 87.0,
        "s1": 183.0, "s2": 103.2, "s3": 70.0, "s4": 3.0, "s5": 3.89, "s6": 88.0,
    },
}

# ==============================================================================
# 6. TABS NAVIGATION
# ==============================================================================
tab_single, tab_batch, tab_analytics, tab_docs = st.tabs([
    "🔮 Single Patient Prediction",
    "📁 Batch Prediction (CSV)",
    "📊 Model Analytics & Benchmarks",
    "ℹ️ Dataset & Feature Guide",
])

# ------------------------------------------------------------------------------
# TAB 1: SINGLE PATIENT PREDICTION
# ------------------------------------------------------------------------------
with tab_single:
    st.subheader("Patient Clinical & Biochemical Parameters")

    # Apply preset if selected
    active_preset = presets_clinical.get(preset, None)

    if input_mode == "Clinical Units (Recommended)":
        st.info("💡 Values are presented in standard clinical units and automatically standardized for the model pipeline.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 👤 Demographics & Vitals")
            def_age = float(active_preset["age"]) if active_preset else 50.0
            age = st.slider("Age (years)", min_value=18.0, max_value=85.0, value=def_age, step=1.0)

            def_sex = active_preset["sex"] if active_preset else "Male"
            sex_choice = st.selectbox("Sex", ["Female", "Male"], index=0 if def_sex == "Female" else 1)
            # In raw sklearn diabetes dataset: 1 = female, 2 = male
            sex_numeric = 1.0 if sex_choice == "Female" else 2.0

            def_bmi = float(active_preset["bmi"]) if active_preset else 26.4
            bmi = st.number_input(
                "BMI - Body Mass Index (kg/m²)",
                min_value=15.0, max_value=55.0, value=def_bmi, step=0.1,
                help="Normal: 18.5 - 24.9, Overweight: 25 - 29.9, Obese: ≥ 30",
            )

            def_bp = float(active_preset["bp"]) if active_preset else 95.0
            bp = st.number_input(
                "Blood Pressure - MAP (mm Hg)",
                min_value=60.0, max_value=140.0, value=def_bp, step=0.5,
                help="Mean Arterial Pressure ≈ DBP + 1/3(SBP - DBP). Normal resting MAP is ~70-100 mm Hg.",
            )

        with col2:
            st.markdown("#### 🧪 Lipid Profile (Serum)")
            def_s1 = float(active_preset["s1"]) if active_preset else 189.0
            s1 = st.number_input(
                "s1: Total Cholesterol (mg/dL)",
                min_value=80.0, max_value=350.0, value=def_s1, step=1.0,
                help="Desirable: < 200 mg/dL",
            )

            def_s2 = float(active_preset["s2"]) if active_preset else 115.0
            s2 = st.number_input(
                "s2: LDL Cholesterol (mg/dL)",
                min_value=30.0, max_value=260.0, value=def_s2, step=1.0,
                help="Optimal: < 100 mg/dL",
            )

            def_s3 = float(active_preset["s3"]) if active_preset else 50.0
            s3 = st.number_input(
                "s3: HDL Cholesterol (mg/dL)",
                min_value=20.0, max_value=110.0, value=def_s3, step=1.0,
                help="Protective/Good cholesterol. > 50 mg/dL for women, > 40 mg/dL for men.",
            )

            def_s4 = float(active_preset["s4"]) if active_preset else 4.0
            s4 = st.number_input(
                "s4: Total / HDL Ratio",
                min_value=1.5, max_value=11.0, value=def_s4, step=0.1,
                help="Optimal ratio is below 3.5 - 4.0",
            )

        with col3:
            st.markdown("#### 🔬 Metabolic Markers")
            def_s5 = float(active_preset["s5"]) if active_preset else 4.64
            s5 = st.number_input(
                "s5: Log Serum Triglycerides",
                min_value=3.0, max_value=6.5, value=def_s5, step=0.01,
                help="Logarithm of serum triglycerides level (e.g. ln(100 mg/dL) ≈ 4.6). Strongest positive driver in the model.",
            )

            def_s6 = float(active_preset["s6"]) if active_preset else 91.0
            s6 = st.number_input(
                "s6: Blood Glucose (mg/dL)",
                min_value=50.0, max_value=160.0, value=def_s6, step=1.0,
                help="Fasting blood sugar. Normal: 70 - 99 mg/dL.",
            )

        clinical_inputs = {
            "age": age, "sex": sex_numeric, "bmi": bmi, "bp": bp,
            "s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6,
        }
        normalized_inputs = clinical_to_normalized(clinical_inputs)

    else:
        st.info("💡 You are entering raw mean-centered normalized feature values directly (sklearn format: typical range -0.15 to +0.20).")
        col1, col2 = st.columns(2)
        normalized_inputs = {}

        for idx, col_name in enumerate(FEATURE_COLS):
            target_col = col1 if idx < 5 else col2
            with target_col:
                normalized_inputs[col_name] = st.slider(
                    f"{col_name} - {FEATURE_LABELS[col_name]}",
                    min_value=-0.15, max_value=0.20, value=0.0, step=0.005,
                    format="%.3f",
                )
        clinical_inputs = normalized_to_clinical(normalized_inputs)

    # Convert to single-row DataFrame
    input_df = pd.DataFrame([normalized_inputs])

    st.write("---")
    predict_clicked = st.button("🔮 Compute Disease Progression Prediction", type="primary", use_container_width=True)

    # Perform prediction
    pred_val = float(model.predict(input_df)[0])
    risk_label, risk_class, risk_desc = get_risk_assessment(pred_val)

    # Calculate percentile against training target distribution
    clipped_pred = np.clip(pred_val, TARGET_STATS["min"], TARGET_STATS["max"])
    pct_rank = round(((clipped_pred - TARGET_STATS["min"]) / (TARGET_STATS["max"] - TARGET_STATS["min"])) * 100, 1)

    st.markdown("### Prediction Results")

    res_col1, res_col2 = st.columns([1, 1.3])

    with res_col1:
        st.markdown(
            f"""
            <div class="prediction-card">
                <div style="font-size: 0.95rem; color: #64748b; font-weight: 600; text-transform: uppercase;">
                    Quantitative Disease Progression Index
                </div>
                <div style="font-size: 3.5rem; font-weight: 800; color: #1e3a8a; margin: 8px 0;">
                    {pred_val:.1f}
                </div>
                <span class="risk-badge {risk_class}">{risk_label}</span>
                <div style="margin-top: 16px; font-size: 0.9rem; color: #475569; text-align: left;">
                    <strong>Clinical Note:</strong> {risk_desc}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with res_col2:
        st.markdown("#### 📈 Population Comparison Gauge")
        st.write(
            f"The predicted progression index of **{pred_val:.1f}** ranks at approximately the **{pct_rank}th percentile** "
            f"of the baseline diabetes cohort (mean: {TARGET_STATS['mean']:.1f}, range: {int(TARGET_STATS['min'])} – {int(TARGET_STATS['max'])})."
        )

        st.progress(pct_rank / 100.0)

        # Scale reference legend
        scale_df = pd.DataFrame({
            "Risk Tier": ["Mild (< 100)", "Moderate (100 - 180)", "Accelerated (> 180)"],
            "Target Range": ["25.0 - 100.0", "100.0 - 180.0", "180.0 - 346.0"],
            "Cohort Proportion": ["~27% of patients", "~42% of patients", "~31% of patients"],
        })
        st.table(scale_df)

    # Feature Contribution Analysis
    st.write("---")
    st.subheader("🔍 Feature Impact Breakdown (Model Explainability)")
    st.caption("How each clinical variable pushed this patient's prediction above or below the baseline average:")

    contrib_df, intercept = compute_feature_contributions(model, input_df)

    if contrib_df is not None:
        chart_col1, chart_col2 = st.columns([1.4, 1])

        with chart_col1:
            chart_data = contrib_df.copy()
            chart_data["Impact Direction"] = chart_data["Contribution"].apply(lambda x: "Increases Progression" if x >= 0 else "Decreases Progression")

            st.bar_chart(
                data=chart_data.set_index("Code")[["Contribution"]],
                use_container_width=True,
            )

        with chart_col2:
            st.dataframe(
                contrib_df[["Code", "Feature", "Contribution", "Coefficient"]].style.format({
                    "Contribution": "{:+.2f}",
                    "Coefficient": "{:.2f}",
                }),
                height=300,
                use_container_width=True,
            )
            st.caption(f"Baseline Pipeline Intercept: **{intercept:.2f}**")

# ------------------------------------------------------------------------------
# TAB 2: BATCH PREDICTION (CSV)
# ------------------------------------------------------------------------------
with tab_batch:
    st.subheader("📁 Batch Patient Scoring via CSV")
    st.write("Upload a CSV file containing multiple patient records to generate batch disease progression predictions.")

    # Download Template Button
    sample_template = pd.DataFrame([
        {
            "age": 48.0, "sex": 1.0, "bmi": 24.5, "bp": 90.0,
            "s1": 175.0, "s2": 105.0, "s3": 55.0, "s4": 3.2, "s5": 4.25, "s6": 88.0,
        },
        {
            "age": 62.0, "sex": 2.0, "bmi": 31.8, "bp": 108.0,
            "s1": 215.0, "s2": 140.0, "s3": 38.0, "s4": 5.6, "s5": 5.10, "s6": 112.0,
        },
        {
            "age": 35.0, "sex": 1.0, "bmi": 22.0, "bp": 82.0,
            "s1": 160.0, "s2": 95.0, "s3": 60.0, "s4": 2.7, "s5": 3.90, "s6": 82.0,
        },
    ])

    template_csv = sample_template.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Sample CSV Template",
        data=template_csv,
        file_name="diabetes_batch_sample.csv",
        mime="text/csv",
    )

    batch_units = st.radio(
        "Units used in uploaded CSV:",
        ["Clinical Units (Age in years, BMI in kg/m², mg/dL)", "Normalized Sklearn Values"],
        horizontal=True,
    )

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(batch_df)}** records:")
            st.dataframe(batch_df.head(), use_container_width=True)

            # Check required columns
            missing_cols = [c for c in FEATURE_COLS if c not in batch_df.columns]

            if missing_cols:
                st.error(f"❌ Missing required columns in CSV: `{missing_cols}`. Expected columns: `{FEATURE_COLS}`")
            else:
                if st.button("🚀 Run Batch Prediction", type="primary"):
                    with st.spinner("Calculating predictions..."):
                        if batch_units.startswith("Clinical"):
                            # Convert each row to normalized values
                            norm_rows = []
                            for _, r in batch_df.iterrows():
                                norm_rows.append(clinical_to_normalized(r[FEATURE_COLS].to_dict()))
                            eval_df = pd.DataFrame(norm_rows)
                        else:
                            eval_df = batch_df[FEATURE_COLS].copy()

                        predictions = model.predict(eval_df)
                        result_df = batch_df.copy()
                        result_df["Predicted_Progression"] = np.round(predictions, 2)
                        result_df["Risk_Category"] = result_df["Predicted_Progression"].apply(
                            lambda x: get_risk_assessment(x)[0].split(" / ")[0]
                        )

                        st.success(f"Successfully processed {len(result_df)} patients!")

                        # Summary metrics
                        sc1, sc2, sc3 = st.columns(3)
                        with sc1:
                            st.metric("Average Progression", f"{result_df['Predicted_Progression'].mean():.2f}")
                        with sc2:
                            st.metric("Min Progression", f"{result_df['Predicted_Progression'].min():.2f}")
                        with sc3:
                            st.metric("Max Progression", f"{result_df['Predicted_Progression'].max():.2f}")

                        st.markdown("#### Predictions Table")
                        st.dataframe(result_df, use_container_width=True)

                        # CSV Download of Results
                        csv_out = result_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "📥 Download Predictions as CSV",
                            data=csv_out,
                            file_name="diabetes_predictions_output.csv",
                            mime="text/csv",
                        )

        except Exception as e:
            st.error(f"Error parsing uploaded file: {e}")

# ------------------------------------------------------------------------------
# TAB 3: MODEL ANALYTICS & BENCHMARKS
# ------------------------------------------------------------------------------
with tab_analytics:
    st.subheader("📊 Model Training & Benchmarking Analysis")
    st.caption("Comprehensive results from experimental evaluation and hyperparameter tuning in `diabetes_regression_project.ipynb`.")

    col_bench1, col_bench2 = st.columns([1.5, 1])

    with col_bench1:
        st.markdown("#### 14 Model Cross-Validation Comparison")
        # Exact values from notebook cell 38
        benchmark_data = [
            {"Model": "Ridge (Selected)", "CV R²": 0.4808, "CV RMSE": 55.3742, "CV MAE": 44.9989, "Test R²": 0.4541, "Test RMSE": 53.7775, "Test MAE": 42.8120},
            {"Model": "Lasso", "CV R²": 0.4808, "CV RMSE": 55.3885, "CV MAE": 45.1358, "Test R²": 0.4669, "Test RMSE": 53.1467, "Test MAE": 42.8030},
            {"Model": "Linear Regression", "CV R²": 0.4804, "CV RMSE": 55.3946, "CV MAE": 45.0482, "Test R²": 0.4526, "Test RMSE": 53.8534, "Test MAE": 42.7941},
            {"Model": "Bayesian Ridge", "CV R²": 0.4794, "CV RMSE": 55.4564, "CV MAE": 45.1015, "Test R²": 0.4579, "Test RMSE": 53.5905, "Test MAE": 42.8704},
            {"Model": "Huber Regressor", "CV R²": 0.4730, "CV RMSE": 55.7427, "CV MAE": 45.2963, "Test R²": 0.4468, "Test RMSE": 54.1373, "Test MAE": 42.8797},
            {"Model": "ElasticNet", "CV R²": 0.4584, "CV RMSE": 56.6585, "CV MAE": 46.8668, "Test R²": 0.4548, "Test RMSE": 53.7467, "Test MAE": 44.0718},
            {"Model": "Extra Trees", "CV R²": 0.4290, "CV RMSE": 58.0833, "CV MAE": 47.2976, "Test R²": 0.4685, "Test RMSE": 53.0679, "Test MAE": 43.9224},
            {"Model": "Random Forest", "CV R²": 0.3983, "CV RMSE": 59.4871, "CV MAE": 48.5263, "Test R²": 0.4415, "Test RMSE": 54.3984, "Test MAE": 44.1712},
            {"Model": "AdaBoost", "CV R²": 0.3920, "CV RMSE": 59.9838, "CV MAE": 50.0531, "Test R²": 0.4419, "Test RMSE": 54.3774, "Test MAE": 45.0762},
            {"Model": "Gradient Boosting", "CV R²": 0.3654, "CV RMSE": 61.1980, "CV MAE": 49.6144, "Test R²": 0.4551, "Test RMSE": 53.7307, "Test MAE": 44.5194},
            {"Model": "Bagging Regressor", "CV R²": 0.3492, "CV RMSE": 61.7694, "CV MAE": 49.8228, "Test R²": 0.3783, "Test RMSE": 57.3941, "Test MAE": 46.6225},
            {"Model": "K-Nearest Neighbors", "CV R²": 0.3447, "CV RMSE": 62.1104, "CV MAE": 49.4423, "Test R²": 0.4248, "Test RMSE": 55.2037, "Test MAE": 42.7775},
            {"Model": "Support Vector Regression", "CV R²": 0.1003, "CV RMSE": 73.0626, "CV MAE": 61.1362, "Test R²": 0.1822, "Test RMSE": 65.8235, "Test MAE": 56.0295},
            {"Model": "Decision Tree", "CV R²": -0.1240, "CV RMSE": 81.0413, "CV MAE": 62.2734, "Test R²": 0.0776, "Test RMSE": 69.9071, "Test MAE": 53.9213},
        ]
        bench_df = pd.DataFrame(benchmark_data)
        st.dataframe(bench_df.set_index("Model"), height=420, use_container_width=True)

    with col_bench2:
        st.markdown("#### Permutation Feature Importance")
        st.caption("Drop in validation R² score when feature values are randomly shuffled (20 repeats):")
        # Computed via sklearn.inspection.permutation_importance
        perm_data = pd.DataFrame({
            "Feature": ["s5 (Triglycerides)", "s1 (Total Chol)", "bmi (BMI)", "s2 (LDL)", "sex", "bp (Blood Pressure)", "s4 (TCH)", "s3 (HDL)", "s6 (Glucose)", "age"],
            "Mean R² Drop": [0.4101, 0.3413, 0.2223, 0.1322, 0.0733, 0.0498, 0.0300, 0.0165, 0.0035, -0.0033],
        })
        st.bar_chart(perm_data.set_index("Feature"), use_container_width=True)

    st.write("---")
    st.markdown("#### 🎯 Hyperparameter Tuning & Pipeline Specifications")
    st.markdown(
        """
        - **Pipeline Preprocessing:** `StandardScaler()` applied prior to regression fitting.
        - **Best Estimator:** `Ridge(alpha=1.0, solver='lsqr', random_state=42)`
        - **Validation Strategy:** 5-Fold Cross-Validation with shuffle (`KFold(n_splits=5, shuffle=True, random_state=42)`).
        - **Why Ridge Regressor Won:** The 6 blood serum measurements (`s1` through `s6`) exhibit severe multicollinearity (VIF values up to 59+ in raw correlations). L2 regularization efficiently shrinks collinear coefficients, producing the most stable generalization without severe overfitting.
        """
    )

# ------------------------------------------------------------------------------
# TAB 4: DATASET & FEATURE DOCUMENTATION
# ------------------------------------------------------------------------------
with tab_docs:
    st.subheader("ℹ️ Dataset & Clinical Feature Glossary")

    st.markdown(
        """
        ### About the Scikit-Learn Diabetes Dataset
        The dataset consists of **442 diabetes patients** initially examined in the landmark study by **Efron, Hastie, Johnstone, and Tibshirani (2004)**:
        *Least Angle Regression, Annals of Statistics (with discussion), 407-499*.

        #### Target Variable
        - **`target`**: A quantitative measure of **disease progression one year after baseline** (range: 25 to 346, mean: 152.1, standard deviation: 77.1).
        Higher values correspond to faster, more severe pathological deterioration of diabetes markers over a 12-month window.

        ---
        ### Baseline Predictor Variables (10 Features)
        """
    )

    doc_data = [
        {"Feature": "age", "Name": "Age", "Units": "Years (19 - 79)", "Clinical Significance": "Biological age of patient at baseline screening."},
        {"Feature": "sex", "Name": "Sex", "Units": "Encoded (Female / Male)", "Clinical Significance": "Biological sex factor."},
        {"Feature": "bmi", "Name": "Body Mass Index", "Units": "kg/m² (18.0 - 42.2)", "Clinical Significance": "Adiposity measure; high BMI correlates strongly with insulin resistance."},
        {"Feature": "bp", "Name": "Mean Arterial Pressure", "Units": "mm Hg (62.0 - 133.0)", "Clinical Significance": "Average arterial blood pressure throughout one cardiac cycle."},
        {"Feature": "s1", "Name": "Total Serum Cholesterol (tc)", "Units": "mg/dL (97.0 - 301.0)", "Clinical Significance": "Circulating blood cholesterol level."},
        {"Feature": "s2", "Name": "Low-Density Lipoproteins (ldl)", "Units": "mg/dL (41.6 - 242.4)", "Clinical Significance": "Atherogenic lipoprotein concentration."},
        {"Feature": "s3", "Name": "High-Density Lipoproteins (hdl)", "Units": "mg/dL (22.0 - 99.0)", "Clinical Significance": "Protective, anti-atherogenic cholesterol carrier."},
        {"Feature": "s4", "Name": "Total Cholesterol / HDL (tch)", "Units": "Ratio (2.0 - 9.09)", "Clinical Significance": "Cardiometabolic risk ratio."},
        {"Feature": "s5", "Name": "Log Serum Triglycerides (ltg)", "Units": "Log level (3.26 - 6.11)", "Clinical Significance": "Key indicator of hypertriglyceridemia and metabolic syndrome."},
        {"Feature": "s6", "Name": "Blood Sugar / Glucose (glu)", "Units": "mg/dL (58.0 - 124.0)", "Clinical Significance": "Fasting blood sugar level at baseline."},
    ]
    st.table(pd.DataFrame(doc_data))

    st.warning(
        "⚠️ **Disclaimer:** This tool is intended for research, demonstration, and clinical decision-support reference only. "
        "It does not replace professional medical advice, clinical diagnosis, or individualized treatment plans from licensed healthcare providers."
    )

