# Telco Customer Churn Prediction

This project predicts customer churn for a telecom company using **XGBoost**. It includes data preprocessing, feature engineering, model training, and professional visualization of model performance.

---

## **Dataset**

The dataset used is `Telco_customer_churn.xlsx`, which includes:

- Customer demographic information (gender, age, etc.)
- Account information (tenure, services subscribed, payment method, etc.)
- Churn indicator (`Churn Value`)

> **Note:** Sensitive columns like `CustomerID`, `City`, `Latitude/Longitude`, and others were removed during preprocessing.

---

## **Project Workflow**

1. **Data Preprocessing**
   - Drop irrelevant columns.
   - Handle numeric columns (`Total Charges`) and zero-tenure ratio.
   - Encode categorical variables with **One-Hot Encoding** (limited to 2-4 unique values per column).
   - Feature engineering:
     - `Monthly Tenure Ratio` = Monthly Charges / Tenure Months
     - `Zip_Churn_Rate` = average churn rate per zip code.

2. **Data Splitting**
   - Split into training and test sets (90% train, 10% test) using stratified sampling to preserve churn class distribution.

3. **Model Training**
   - XGBoost classifier with tuned hyperparameters:
     ```python
     params = {
         'scale_pos_weight': 1.1,
         'n_estimators': 1500,
         'learning_rate': 0.005,
         'max_depth': 4,
         'min_child_weight': 12,
         'gamma': 0.2,
         'subsample': 0.9,
         'colsample_bytree': 0.9,
         'reg_alpha': 0.005,
         'reg_lambda': 1,
         'objective': 'binary:logistic',
         'tree_method': 'hist',
         'random_state': 42
     }
     ```

4. **Evaluation Metrics**
   - **Classification Report** (precision, recall, F1-score per class) visualized as a **heatmap**.
   - **ROC Curve** with AUC for overall model performance.

---

## **Results**

### Classification Report Heatmap

![Classification Report Heatmap](classification_report_heatmap.png)

### ROC Curve

![ROC Curve](roc_curve.png)

---
