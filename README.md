
# 🏦 Bank Term Deposit Subscription Prediction

## 📝 Overview
This repository contains a machine learning project aimed at predicting whether a customer will subscribe to a term deposit offered by a Portuguese banking institution. The project is based on direct marketing campaign data collected through phone calls. The goal is to build a predictive model and provide clear explanations of how the model works, ensuring transparency and interpretability for stakeholders.

## 🎯 Problem Statement
The task is to predict whether a customer will open a term deposit based on their demographic and behavioral data. 

## 📊 Dataset
The dataset used in this project is `bank-additional-full.csv`, which contains 41,188 examples and 20 input features. The data is related to direct marketing campaigns conducted via phone calls by a Portuguese banking institution between May 2008 and November 2010. The features include customer demographics, campaign-related attributes, and macroeconomic indicators.

The dataset is available on Kaggle: [Bank Additional Full Dataset](https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv/data).

### 🔑 Key Features:
- **Demographics**: Age, job, marital status, education, etc.
- **Campaign-related**: Contact type, month, day of the week, duration, etc.
- **Economic Indicators**: Employment variation rate, consumer price index, etc.
- **Target Variable**: `y` (whether the customer subscribed to a term deposit).

## 🚀 Project Goals
1. **Model Development**: Build a machine learning model to predict the likelihood of a customer subscribing to a term deposit.
2. **Model Interpretability**: Provide clear explanations of the model's predictions using techniques like SHAP and misclassification analysis.

## 📂 Repository Structure
```
bank-term-deposit-prediction/
├── data/                   # Folder containing the dataset
│   └── bank-additional-full.csv
│   └── bank-additional-full-clean.csv
├── notebooks/              # Jupyter notebooks for EDA, modeling, and explanation
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_interpretability.ipynb
├── src/                    # Source code for preprocessing, modeling, and evaluation
│   ├── process_bank_churn.py
│   ├── viz_plots.py
├── models/                 # Saved models

```

## 🔍 Key Steps
1. **Exploratory Data Analysis (EDA)**: Analyze the dataset to understand feature distributions, correlations, and missing values.
2. **Data Preprocessing**: Handle missing values, encode categorical variables, and scale numerical features.
3. **Modeling**: Train and evaluate machine learning models (e.g., Logistic Regression, Random Forest, XGBoost) using appropriate metrics to evaluate them.
4. **Interpretability**: Use explainability tools to provide insights into the model's predictions.


## 📈 Results
The final model achieves an F1-score of 51.76% on the validation dataset. The interpretability analysis reveals that features like economic factors are the most influential in predicting term deposit subscriptions.

| Model                     |  Validation F1 | Validation Average Precision | Train Threshold | Train F1 | Train Average Precision |
|---------------------------|--------------|----------------------------|----------------|---------|------------------------|
| Logistic Regression       |  49.69        | 46.96                      | 0.65         | 49.02   | 44.24                  |
| KNN                       |  48.59        | 43.13                      | 0.25          | 51.19   | 48.85                  |
| Decision Tree             |  50.37        | 43.22                      | 0.45          | 49.53   | 41.53                  |
| XGBoost                   |  44.00        | 43.48                      | 0.60          | 71.24   | 75.27                  |
| XGBoost Hyperopt          | **51.76**        | **49.11**                      | 0.60          | 57.99   | 60.47                  |
| XGBoost Random Search     |  50.08        | 49.39                      | 0.30          | 56.77   | 61.20                  |
| Random Forest             |  46.36        | 49.33                      | 0.40          | 60.11   | 67.91                  |
| Balanced Bagging          |  47.00        | 48.18                      | 0.75         | 61.87   | 71.77                  |



## 🔮 Future Work
- Implement a Streamlit dashboard for interactive visualization of predictions and model explanations.


