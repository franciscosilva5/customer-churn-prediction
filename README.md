# Customer Churn Prediction

Machine Learning project for predicting customer churn in a telecommunications company.

The project covers the full workflow from exploratory data analysis to a deployed Streamlit application.

## Demo

The Streamlit application allows users to enter customer information and receive a churn probability and risk classification.

![Customer Churn Prediction App](images/app-screenshot.png)

## Business Problem

Customer churn represents customers who stop using a company's services.

The objective of this project is to identify customers with a higher probability of churn so that a company can prioritize retention actions.

## Dataset

IBM Telco Customer Churn dataset.

* 7,043 customers
* 21 original variables
* Target: `Churn`
* Churn rate: approximately 26.5%

Examples of features:

* Contract type
* Tenure
* Monthly charges
* Internet service
* Payment method
* Technical support
* Online security

## Exploratory Data Analysis

Several strong patterns were identified.

### Contract

Churn rate by contract:

* Month-to-month: 42.7%
* One year: 11.3%
* Two year: 2.8%

Long-term contracts were strongly associated with lower churn.

### Internet Service

Churn rate:

* Fiber optic: 41.9%
* DSL: 19.0%
* No internet: 7.4%

### Tenure

Average tenure:

* Customers who stayed: 37.6 months
* Customers who churned: 18.0 months

### Monthly Charges

Average monthly charges:

* Customers who stayed: 61.27
* Customers who churned: 74.44

## Data Preparation

The preprocessing pipeline includes:

* Cleaning `TotalCharges`
* Handling missing values
* Removing `customerID`
* One-hot encoding categorical variables
* Standardizing numerical variables
* Train/test split with stratification

The preprocessing and model are combined in a Scikit-learn `Pipeline`.

## Models Tested

### Logistic Regression

ROC-AUC:

`0.842`

Cross-validation ROC-AUC:

`0.845 ± 0.011`

### Random Forest

ROC-AUC:

`0.827`

Logistic Regression produced the strongest overall performance and was selected as the final model.

## Threshold Optimization

The default classification threshold of 0.50 was not ideal for detecting churn.

A threshold of `0.39` was selected as a better balance between precision and recall.

At this threshold:

* Precision: 57%
* Recall: 68%
* F1-score: 62%

Confusion matrix:

* True negatives: 841
* False positives: 194
* False negatives: 120
* True positives: 254

This means the model detects approximately 68% of customers who actually churn.

## Model Interpretation

Some of the strongest positive churn indicators were:

* Fiber optic internet
* Electronic check payment
* Paperless billing
* Multiple lines

Some of the strongest indicators associated with customer retention were:

* Two-year contracts
* One-year contracts
* Online security
* Technical support

## Streamlit Application

The model is integrated into a Streamlit application.

Users can enter customer characteristics and receive a real-time churn probability.

Example:

`Churn probability: 76.6%`

`High churn risk`

## Project Structure

```text
customer-churn/
├── app.py
├── churn_model.joblib
├── churn_threshold.joblib
├── exploration.ipynb
├── requirements.txt
├── README.md
├── images/
│   └── app-screenshot.png
└── data/
    └── raw/
        └── customer_churn.csv
```

## Run Locally

Clone the repository:

```bash
git clone https://github.com/Vercetius/customer-churn-prediction.git
cd customer-churn-prediction
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the app:

```bash
python -m streamlit run app.py
```

## Technologies

* Python
* pandas
* NumPy
* Scikit-learn
* Streamlit
* Jupyter Notebook
* Git
* GitHub

## Future Improvements

Possible next steps:

* Compare additional models such as XGBoost or LightGBM
* Hyperparameter tuning
* SHAP explainability
* Cost-sensitive threshold optimization
* Deploy the Streamlit application publicly
* Add automated tests

## App Screenshots

### Customer Input

![App Perguntas](images/app-perguntas.png)

### Prediction Result

![App Resultado](images/app-resultado.png)
