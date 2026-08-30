# -Telecom-churn-prediction
Data science and Machine Learning project for Telecom churn prediction
Predicting which telecom customers are likely to churn, using the IBM/Kaggle Telco Customer Churn dataset (7,043 customers, 21 columns).

# Problem

Telecom companies lose revenue when customers leave. This project builds a classifier that flags customers at high risk of churning, so a business could target retention offers instead of guessing.

# Dataset
Source: Telco Customer Churn (Kaggle)
7,043 rows, 21 columns, one row per customer
Target: Churn (Yes/No)
Features cover demographics, account info (tenure, contract type, payment method), and subscribed services (internet, phone, streaming, tech support, etc.)
Pipeline

The notebook runs top to bottom as a single linear pipeline. No step is optional — later steps depend on the output of earlier ones.

# 1. Load data
Read WA_Fn-UseC_-Telco-Customer-Churn.csv into a DataFrame
Confirm shape: 7,043 rows × 21 columns, one row per customer
# 2. Clean
Check and drop duplicate rows
Drop customerID — it's an identifier, not a predictor
TotalCharges was stored as a string because a handful of rows (new customers with 0 tenure) had blank values instead of numbers. Converted to numeric with pd.to_numeric(..., errors="coerce"), then filled the resulting NaNs with 0
Mapped the target Churn from Yes/No to 1/0
# 3. Exploratory Data Analysis (EDA)
Computed overall churn rate
Grouped churn rate by Contract and by InternetService to see which segments churn most
Plotted:
Churn distribution (pie chart) → churn_pie.png
Churn rate for all 16 categorical features in one grid → churn_by_category.png
Stacked bar counts (churned vs not) for the four features most linked to churn — contract, internet service, payment method, tech support → churn_counts.png
# 4. Feature engineering

Four new features, all derived from existing columns:

tenure_group: tenure binned into 0-1yr, 1-2yr, 2-4yr, 4-5yr, 5yr+
num_services: count of "Yes" across the 6 add-on service columns (online security, online backup, device protection, tech support, streaming TV, streaming movies)
avg_monthly_spend: TotalCharges / tenure (tenure=0 replaced with 1 to avoid divide-by-zero)
no_support_services: binary flag for customers with neither online security nor tech support — a common churn-risk segment
# 5. Preprocessing
Split features (X) and target (y)
One-hot encoded all categorical columns with pd.get_dummies(drop_first=True)
Dropped TotalCharges after engineering — it's ≈ tenure × MonthlyCharges, so keeping it alongside avg_monthly_spend would introduce multicollinearity
Final feature count printed after encoding
# 6. Train/test split + scaling + resampling
80/20 train-test split, stratified on the target so both sets keep the same churn ratio
Standardized features with StandardScaler (fit on train, applied to test — no leakage)
Applied SMOTE only to the training set, after the split — the test set stays as real, untouched, imbalanced data, so test metrics reflect real-world performance, not oversampled performance
# 7. Model comparison
Trained three models on the SMOTE-balanced training data: Logistic Regression, Random Forest, Gradient Boosting
Evaluated all three on the same untouched test set: accuracy, precision, recall, F1, ROC-AUC
Ranked by ROC-AUC to pick the best candidate for tuning
# 8. Hyperparameter tuning
Ran GridSearchCV on Gradient Boosting over n_estimators (100, 200), learning_rate (0.05, 0.1), max_depth (2, 3, 4)
5-fold cross-validation, scored on ROC-AUC
Extracted the best estimator
# 9. Final evaluation
Scored the tuned model on the held-out test set: accuracy, precision, recall, F1, ROC-AUC
Ran a separate 5-fold CV on the original (non-resampled) training data as a sanity check against the SMOTE-trained numbers
# 10. Feature importance
Pulled feature_importances_ from the tuned Gradient Boosting model
Reported the top 12 features driving churn predictions
Results

Final tuned Gradient Boosting model, evaluated on the held-out test set:

Metric	Score
ROC-AUC	0.843
F1	0.62
Recall	0.72
Precision	0.54
5-fold CV ROC-AUC	0.846 ± 0.014
