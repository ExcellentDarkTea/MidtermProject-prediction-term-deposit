
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
2. **Model Interpretability**: Provide clear explanations of the model's predictions using techniques like SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations).

## 📂 Repository Structure
```
bank-term-deposit-prediction/
├── data/                   # Folder containing the dataset
│   └── bank-additional-full.csv
├── notebooks/              # Jupyter notebooks for EDA, modeling, and explanation
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_interpretability.ipynb
├── src/                    # Source code for preprocessing, modeling, and evaluation
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── models/                 # Saved models and artifacts
├── requirements.txt        # Python dependencies
├── README.md               # Project overview and instructions
└── LICENSE                 # License information
```

## 🔍 Key Steps
1. **Exploratory Data Analysis (EDA)**: Analyze the dataset to understand feature distributions, correlations, and missing values.
2. **Data Preprocessing**: Handle missing values, encode categorical variables, and scale numerical features.
3. **Modeling**: Train and evaluate machine learning models (e.g., Logistic Regression, Random Forest, XGBoost) using appropriate metrics to evaluate them.
4. **Interpretability**: Use explainability tools to provide insights into the model's predictions.



## 📈 Results
The final model achieves an accuracy of X% with an F1-score of Y%. The interpretability analysis reveals that features like `duration`, `age`, and `job` are the most influential in predicting term deposit subscriptions.

## 🔮 Future Work
- Implement a Streamlit dashboard for interactive visualization of predictions and model explanations.
- Prepare the model for deployment by creating a pipeline.

