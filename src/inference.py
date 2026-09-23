from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def load_model_artifacts():
    model = joblib.load(
        MODELS_DIR / "churn_model.joblib"
    )

    threshold = float(
        joblib.load(
            MODELS_DIR / "churn_threshold.joblib"
        )
    )

    return model, threshold


def build_customer_dataframe(data):
    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in data
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    return pd.DataFrame(
        [
            {
                column: data[column]
                for column in FEATURE_COLUMNS
            }
        ]
    )


def predict_churn(
    data,
    model,
    threshold,
):
    customer = build_customer_dataframe(
        data
    )

    probability = float(
        model.predict_proba(
            customer
        )[0, 1]
    )

    return {
        "probability": probability,
        "threshold": float(threshold),
        "high_risk": (
            probability >= threshold
        ),
    }
