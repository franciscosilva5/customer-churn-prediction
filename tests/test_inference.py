import pytest

from src.inference import (
    FEATURE_COLUMNS,
    build_customer_dataframe,
    load_model_artifacts,
    predict_churn,
)


SAMPLE_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 1020.0,
}


def test_model_artifacts_load():
    model, threshold = load_model_artifacts()

    assert hasattr(
        model,
        "predict_proba",
    )

    assert 0 < threshold < 1


def test_customer_dataframe_schema():
    dataframe = build_customer_dataframe(
        SAMPLE_CUSTOMER
    )

    assert list(
        dataframe.columns
    ) == FEATURE_COLUMNS

    assert len(dataframe) == 1


def test_missing_feature_rejected():
    invalid = dict(
        SAMPLE_CUSTOMER
    )

    invalid.pop(
        "TotalCharges"
    )

    with pytest.raises(
        ValueError
    ):
        build_customer_dataframe(
            invalid
        )


def test_prediction_output():
    model, threshold = load_model_artifacts()

    result = predict_churn(
        SAMPLE_CUSTOMER,
        model,
        threshold,
    )

    assert (
        0.0
        <= result["probability"]
        <= 1.0
    )

    assert isinstance(
        result["high_risk"],
        bool,
    )

    assert (
        result["threshold"]
        == threshold
    )
