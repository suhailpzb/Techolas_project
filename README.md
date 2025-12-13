# **📊 Loan Approval Prediction — Machine Learning Project**


This project is an end-to-end machine learning application that predicts whether a loan application should be approved or rejected based on user-provided demographic and financial information.
It includes data preprocessing, feature engineering, model training, evaluation, and a deployable Python script.


## **🚀 Project Overview**

Banks and financial institutions receive thousands of loan applications every day. Manually evaluating each application is time-consuming and inconsistent.
This project solves that problem by building a machine learning model that automatically predicts loan eligibility with high accuracy.


## **🧠 Key Features**

Complete data cleaning & preprocessing pipeline

Outlier detection using IQR method

Exploratory Data Analysis (EDA) with visualizations

Model training using:


Logistic Regression

Decision Tree (Best Performing Model)

Random Forest

Hyperparameter tuning

Finalized model saved as loan_pipeline.pkl

A Python file (loan.py) that loads the trained model and makes predictions

## **📈 Machine Learning Workflow**


1️⃣ Data Preprocessing

Handling missing values,
Encoding categorical features,
Scaling numerical features,
Outlier removal using IQR

2️⃣ Exploratory Data Analysis

Boxplots,
Distribution plots,
Correlation analysis

3️⃣ Model Development

The following models were trained and evaluated:

Model	Status,
Logistic Regression	Trained,
Decision Tree	Trained,
Random Forest	Best model – selected,
The Random Forest model delivered the highest accuracy and stability.

4️⃣ Model Deployment

The model is serialized as loan_pipeline.pkl,
The script loan.py loads the model and predicts loan approval

## **🧪 How to Run the Model**

1. Install dependencies
pip install pandas numpy scikit-learn

2. Run prediction script
python loan.py

3. Inside loan.py, the model automatically loads:
import pickle

model = pickle.load(open("loan_pipeline.pkl", "rb"))

## **📊 Results & Insights**

Dicision tree achieved the highest accuracy

Income, credit history, loan amount, and employment status were strong predictors

Data preprocessing significantly improved model performance

## **🎯 Conclusion**

This project demonstrates the complete lifecycle of a machine learning system — from raw data → cleaned data → trained model → deployed application.
It can easily be integrated into a web application or API to automate real-time loan decision-making.


## **📬 Contact**
If you’d like to discuss improvements or collaborate:

GitHub: https://github.com/suhailpzb

Email: suhailpzb@gmail.com
