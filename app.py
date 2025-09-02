import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# -------------------
# Load Model
# -------------------
MODEL_PATH = Path("model/trained_model.joblib")
st.sidebar.success(f"📌 Using model: `{MODEL_PATH}`")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.sidebar.error(f"❌ Failed to load model: {e}")
    st.stop()

# -------------------
# Config
# -------------------
threshold = 0.5
feature_columns = model.feature_names_in_.tolist() if hasattr(model, "feature_names_in_") else []

# -------------------
# Helpers
# -------------------
def prepare_features(user_inputs: dict):
    df = pd.DataFrame([user_inputs])
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns]

def predict_one(user_inputs: dict):
    df = prepare_features(user_inputs)
    prob = model.predict_proba(df)[0][1]
    prediction = "Fraud" if prob >= threshold else "Not Fraud"
    return prediction, prob

def predict_batch(df: pd.DataFrame):
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]
    probs = model.predict_proba(df)[:, 1]
    preds = (probs >= threshold).astype(int)
    results = pd.DataFrame({
        "provider_id": df.index.astype(str),
        "prediction": ["Fraud" if p else "Not Fraud" for p in preds],
        "probability": probs
    })
    return results

# -------------------
# Heat Bar Function (ATS-style)
# -------------------
def fraud_heatbar(prob_percent: int):
    st.markdown(
        f"""
        <div style="margin-top:20px; margin-bottom:20px;">
            <div style="
                height: 30px;
                width: 100%;
                border-radius: 15px;
                background: linear-gradient(to right, green, yellow, orange, red);
                position: relative;
            ">
                <!-- Probability label -->
                <div style="
                    position: absolute;
                    left: {prob_percent}%;
                    top: -25px;
                    transform: translateX(-50%);
                    font-weight: bold;
                    color: #333;
                ">
                    {prob_percent}%
                </div>
                <!-- Pointer line -->
                <div style="
                    position: absolute;
                    left: {prob_percent}%;
                    top: 0;
                    transform: translateX(-50%);
                    height: 30px;
                    width: 4px;
                    background: black;
                    border-radius: 2px;
                "></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------
# Streamlit Config + Styling
# -------------------
st.set_page_config(
    page_title="Healthcare Fraud Detection",
    page_icon="🚨",
    layout="wide",
)

st.markdown("""
<style>
/* Gradient header */
[data-testid="stHeader"] {
    background: linear-gradient(90deg, #004080, #0066CC);
}
/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #f4f6f9;
}
/* KPI Cards */
.kpi-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    margin: 5px;
}
.kpi-value {
    font-size: 28px;
    font-weight: bold;
    color: #004080;
}
.kpi-label {
    font-size: 14px;
    color: #888;
}
</style>
""", unsafe_allow_html=True)

# -------------------
# App Title
# -------------------
st.title("🚨 Healthcare Fraud Detection Dashboard")
st.markdown(
    "<h4 style='text-align: center; color: #555;'>AI-driven fraud risk assessment for healthcare claims</h4>",
    unsafe_allow_html=True
)

# -------------------
# KPI Cards (Demo values, can be dynamic later)
# -------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="kpi-card"><div class="kpi-value">1,254</div><div class="kpi-label">Total Claims</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="kpi-card"><div class="kpi-value">312</div><div class="kpi-label">Fraudulent Cases</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="kpi-card"><div class="kpi-value">75%</div><div class="kpi-label">Model Accuracy</div></div>', unsafe_allow_html=True)

# -------------------
# Tabs
# -------------------
tab1, tab2 = st.tabs(["📂 Batch Upload (CSV)", "📝 Manual Entry"])

# -------------------
# Batch Upload
# -------------------
with tab1:
    st.subheader("📂 Batch Upload for Multiple Providers")
    file = st.file_uploader("Upload Claims Data (CSV)", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.write("🔍 Preview of Uploaded Data", df.head())
        results = predict_batch(df)

        st.subheader("📊 Fraud Detection Results")
        st.dataframe(results, use_container_width=True)

        # Interactive Fraud Distribution
        fraud_counts = results["prediction"].value_counts().reset_index()
        fraud_counts.columns = ["Prediction", "Count"]

        fig = px.pie(
            fraud_counts,
            values="Count",
            names="Prediction",
            color="Prediction",
            color_discrete_map={"Fraud": "red", "Not Fraud": "green"},
            hole=0.5,
            title="Fraud vs Non-Fraud Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            label="⬇️ Download Predictions as CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="fraud_predictions.csv",
            mime="text/csv",
        )

# -------------------
# Manual Entry
# -------------------
with tab2:
    st.subheader("📝 New Insurance Claim Check")

    with st.form("manual_entry_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🏥 Claim Information")
            provider_id = st.text_input("Provider ID")
            claim_id = st.text_input("Claim ID")
            claim_type = st.selectbox("Claim Type", ["Inpatient", "Outpatient"])
            claim_amount = st.number_input("Claim Amount ($)", min_value=0.0, step=100.0)

        with col2:
            st.markdown("### 🧑 Patient Information")
            patient_age = st.number_input("Patient Age", min_value=0, max_value=120, step=1)
            diagnosis = st.text_input("Diagnosis Code(s)", placeholder="E11, I50, N18...")
            procedures = st.text_input("Procedure Code(s)", placeholder="P123, P456...")

        submitted = st.form_submit_button("🚀 Predict Fraud")

    if submitted:
        # Validation
        missing_fields = []
        if not provider_id:
            missing_fields.append("Provider ID")
        if not claim_id:
            missing_fields.append("Claim ID")
        if claim_amount <= 0:
            missing_fields.append("Claim Amount")
        if patient_age <= 0:
            missing_fields.append("Patient Age")
        if not diagnosis:
            missing_fields.append("Diagnosis Code(s)")
        if not procedures:
            missing_fields.append("Procedure Code(s)")

        if missing_fields:
            st.error(f"⚠️ Please fill in all required fields: {', '.join(missing_fields)}")
        else:
            # Features
            features = {
                "ip_claims_count": 1 if claim_type == "Inpatient" else 0,
                "op_claims_count": 1 if claim_type == "Outpatient" else 0,
                "ip_total_reimbursed": claim_amount if claim_type == "Inpatient" else 0,
                "op_total_reimbursed": claim_amount if claim_type == "Outpatient" else 0,
                "ip_avg_age": patient_age if claim_type == "Inpatient" else 0,
                "op_avg_age": patient_age if claim_type == "Outpatient" else 0,
                "ip_avg_diag_count": len(diagnosis.split(",")) if diagnosis else 0,
                "ip_avg_proc_count": len(procedures.split(",")) if procedures else 0,
            }

            # Prediction
            pred, prob = predict_one(features)
            prob_percent = int(prob*100)

            # Result
            st.success(
                f"✅ Claim **{claim_id}** (Provider: {provider_id}) → "
                f"**{pred}** · Fraud Probability = {prob:.2%}"
            )

            # ATS-style Heat Bar
            fraud_heatbar(prob_percent)
