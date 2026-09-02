from pathlib import Path

import streamlit as st
import pandas as pd
import joblib

# Configuração da página
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent

modelo = joblib.load(BASE_DIR / "models" / "churn_model.joblib")
threshold = joblib.load(BASE_DIR / "models" / "churn_threshold.joblib")

# Cabeçalho
st.title("Customer Churn Prediction")
st.write(
    "Machine Learning application to estimate the probability "
    "that a telecom customer will churn."
)

st.divider()

# Duas colunas
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer information")

    gender = st.selectbox("Gender", ["Female", "Male"])

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with col2:
    st.subheader("Services and billing")

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=700.0
    )

st.divider()

if st.button(
    "Predict churn risk",
    type="primary",
    use_container_width=True
):

    cliente = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    probabilidade = float(
        modelo.predict_proba(cliente)[0, 1]
    )

    st.subheader("Prediction")

    col_prob, col_threshold = st.columns(2)

    with col_prob:
        st.metric(
            "Churn probability",
            f"{probabilidade * 100:.1f}%"
        )

    with col_threshold:
        st.metric(
            "Decision threshold",
            f"{threshold * 100:.0f}%"
        )

    if probabilidade >= threshold:
        st.error("High churn risk")
        st.write(
            "This customer should be considered for a retention action."
        )
    else:
        st.success("Low churn risk")
        st.write(
            "The model considers this customer relatively unlikely to churn."
        )
