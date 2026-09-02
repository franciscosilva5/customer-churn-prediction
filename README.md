# Customer Churn Prediction

Machine Learning project that predicts the probability of customer churn for a telecommunications company.

The project covers the complete ML workflow: exploratory data analysis, preprocessing, model comparison, threshold optimization and deployment through a Streamlit web application.

## Live Demo

Try the application here:

https://customer-churn-prediction-ta-facil-joao.streamlit.app/

The application allows users to enter customer information and receive:

- Churn probability
- Risk classification
- Decision threshold
- Retention recommendation

## App Screenshots

### Customer Input

![Customer Input](images/App%20Perguntas.png)

### Prediction Result

![Prediction Result](images/App%20Resultado.png)

## Business Problem

Customer churn occurs when customers stop using a company's services.

The objective of this project is to identify customers with a higher probability of churn so that a company can prioritize retention actions before those customers leave.

Rather than relying only on overall accuracy, the project focuses particularly on detecting customers who are genuinely at risk of churn.

## Dataset

The project uses the IBM Telco Customer Churn dataset.

- 7,043 customers
- 21 original variables
- Target variable: `Churn`
- Overall churn rate: approximately 26.5%

The dataset includes information such as:

- Contract type
- Customer tenure
- Monthly charges
- Total charges
- Internet service
- Payment method
- Technical support
- Online security
- Streaming services

## Exploratory Data Analysis

Several clear patterns were found before training the models.

### Contract Type

Churn rate by contract:

- Month-to-month: 42.7%
- One year: 11.3%
- Two year: 2.8%

Customers with long-term contracts showed substantially lower churn.

### Internet Service

Churn rate by internet service:

- Fiber optic: 41.9%
- DSL: 19.0%
- No internet: 7.4%

Fiber optic customers showed a considerably higher churn rate in this dataset.

### Customer Tenure

Average tenure:

- Customers who stayed: 37.6 months
- Customers who churned: 18.0 months

Customers with shorter tenure were substantially more likely to churn.

### Monthly Charges

Average monthly charges:

- Customers who stayed: 61.27
- Customers who churned: 74.44

Customers who churned also tended to have higher monthly charges.

## Data Preparation

The preprocessing workflow includes:

- Converting `TotalCharges` to a numerical variable
- Investigating and handling missing values
- Removing `customerID` from the model features
- Separating features and target
- Stratified train/test split
- One-hot encoding categorical features
- Standardizing numerical features

The preprocessing and Logistic Regression model are combined into a Scikit-learn `Pipeline`.

This makes the model reusable because new customer data can be passed directly into the pipeline without manually repeating the preprocessing steps.

## Models Tested

### Logistic Regression

Test ROC-AUC:

`0.842`

5-fold cross-validation ROC-AUC:

`0.845 ± 0.011`

### Random Forest

Test ROC-AUC:

`0.827`

Logistic Regression provided the strongest overall discrimination and was selected as the final model.

## Threshold Optimization

Using the default classification threshold of `0.50` resulted in relatively low recall for churn customers.

Several thresholds were evaluated to study the trade-off between precision and recall.

A threshold of:

`0.39`

was selected as a practical balance between identifying churn customers and limiting false alarms.

At this threshold:

- Precision: approximately 57%
- Recall: approximately 68%
- F1-score: approximately 62%
- ROC-AUC: approximately 0.842

Confusion matrix:

- True negatives: 841
- False positives: 194
- False negatives: 120
- True positives: 254

This means the model detects approximately 68% of customers who actually churn.

## Model Interpretation

The Logistic Regression coefficients were also examined to understand which features influenced the model.

Some of the strongest features associated with higher churn predictions were:

- Fiber optic internet
- Electronic check payment
- Paperless billing
- Multiple lines

Some of the strongest features associated with lower churn predictions were:

- Two-year contracts
- One-year contracts
- Online security
- Technical support

These relationships should be interpreted as associations within the dataset rather than proof of causation.

## Streamlit Application

The trained Scikit-learn pipeline is saved using `joblib` and loaded by the Streamlit application.

The application allows a user to enter the characteristics of a new customer and receive a real-time prediction.

Example output:

`Churn probability: 76.6%`

`High churn risk`

Customers whose predicted probability exceeds the selected threshold are flagged as candidates for retention action.

## Project Structure

```text
customer-churn-prediction/
├── app/
│   └── app.py
├── data/
│   └── raw/
│       └── customer_churn.csv
├── images/
│   ├── App Perguntas.png
│   └── App Resultado.png
├── models/
│   ├── churn_model.joblib
│   └── churn_threshold.joblib
├── notebooks/
│   └── exploration.ipynb
├── .gitignore
├── README.md
└── requirements.txt