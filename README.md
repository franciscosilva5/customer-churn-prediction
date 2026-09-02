# Customer Churn Prediction

Machine Learning project that predicts the probability of customer churn for a telecommunications company.

The project covers the complete ML workflow: exploratory data analysis, preprocessing, model comparison, threshold optimization and deployment through a Streamlit web application.

## Live Demo

Try the application here:

https://customer-churn-ml999.streamlit.app/

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

## Dataset

The project uses the IBM Telco Customer Churn dataset.

- 7,043 customers
- 21 original variables
- Target variable: `Churn`
- Overall churn rate: approximately 26.5%

## Exploratory Data Analysis

### Contract Type

Churn rate by contract:

- Month-to-month: 42.7%
- One year: 11.3%
- Two year: 2.8%

### Internet Service

Churn rate:

- Fiber optic: 41.9%
- DSL: 19.0%
- No internet: 7.4%

### Customer Tenure

Average tenure:

- Customers who stayed: 37.6 months
- Customers who churned: 18.0 months

### Monthly Charges

Average monthly charges:

- Customers who stayed: 61.27
- Customers who churned: 74.44

## Data Preparation

The preprocessing workflow includes:

- Converting `TotalCharges` to numeric
- Handling missing values
- Removing `customerID`
- Stratified train/test split
- One-hot encoding categorical features
- Standardizing numerical features

The preprocessing and Logistic Regression model are combined in a Scikit-learn `Pipeline`.

## Models Tested

### Logistic Regression

- Test ROC-AUC: `0.842`
- 5-fold cross-validation ROC-AUC: `0.845 ± 0.011`

### Random Forest

- Test ROC-AUC: `0.827`

Logistic Regression was selected as the final model.

## Threshold Optimization

A decision threshold of `0.39` was selected.

At this threshold:

- Precision: ~57%
- Recall: ~68%
- F1-score: ~62%

Confusion matrix:

- True negatives: 841
- False positives: 194
- False negatives: 120
- True positives: 254

## Model Interpretation

Features associated with higher churn predictions included:

- Fiber optic internet
- Electronic check payment
- Paperless billing
- Multiple lines

Features associated with lower churn predictions included:

- Two-year contracts
- One-year contracts
- Online security
- Technical support

These are associations in the dataset and should not be interpreted as proof of causation.

## Streamlit Application

The trained model is integrated into a Streamlit application.

Users can enter customer characteristics and receive a real-time churn probability and risk classification.

Example:

`Churn probability: 76.6%`

`High churn risk`

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
```

## Run Locally

Clone the repository:

git clone https://github.com/Vercetius/customer-churn-prediction.git

cd customer-churn-prediction

Create a virtual environment:

python3 -m venv .venv

source .venv/bin/activate

Install dependencies:

python -m pip install -r requirements.txt

Run the app:

python -m streamlit run app/app.py

## Technologies

- Python
- pandas
- NumPy
- Scikit-learn
- Streamlit
- Jupyter Notebook
- Matplotlib
- Git
- GitHub

## Future Improvements

- Hyperparameter optimization
- XGBoost or LightGBM comparison
- SHAP explainability
- Cost-sensitive threshold optimization
- Automated tests
- Model monitoring