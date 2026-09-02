# Customer Churn Prediction

Machine Learning project that predicts the probability of customer churn for a telecommunications company.

## Project Overview

The objective of this project is to identify customers who are at risk of leaving the company.

The project includes:

- Exploratory Data Analysis
- Data cleaning and preprocessing
- Logistic Regression
- Random Forest comparison
- Cross-validation
- Threshold optimization
- ROC-AUC and classification metrics
- Streamlit web application

## Dataset

IBM Telco Customer Churn dataset.

The dataset contains 7,043 customers and 21 original variables.

## Model Performance

Final model: Logistic Regression

- ROC-AUC: ~0.84
- Cross-validation ROC-AUC: ~0.845
- Decision threshold: 0.39

At the selected threshold:

- Churn recall: ~68%
- Churn precision: ~57%
- Churn F1-score: ~62%

## Key Findings

Higher churn risk was associated with:

- Month-to-month contracts
- Fiber optic internet
- Electronic check payment
- Higher monthly charges
- Lower tenure

Long-term contracts were strongly associated with lower churn risk.

## App

The Streamlit application allows a user to enter customer information and receive:

- Churn probability
- Risk classification
- Retention flag

## Run Locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py