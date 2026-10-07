# 📊 Telco Customer Churn Prediction & Interactive Dashboard

A complete end-to-end Machine Learning project designed to predict customer churn, uncover key behavioral drivers, and provide a real-time interactive web application for business stakeholders.

---

## 🚀 Live Demo
You can view the live interactive Streamlit dashboard deployed on Streamlit Community Cloud:  
👉 **[https://m8tbrusyxwkmhl4eyczjhs.streamlit.app/]**

---

## 📌 Project Overview
Customer churn is one of the most critical metrics for subscription-based businesses. Retaining existing customers is significantly more cost-effective than acquiring new ones. This project builds a predictive classification model using the classic Telco Customer Churn dataset to identify high-risk customers, allowing businesses to take proactive retention measures.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python 
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest, Preprocessing, Metrics)
* **Data Visualization:** Matplotlib, Seaborn
* **Interactive App Deployment:** Streamlit

---

## 🔄 Project Workflow

### 1. Data Loading & Inspection
* **Source Dataset:** IBM / Kaggle Telco Customer Churn dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).
* Loaded data using Pandas and performed initial inspections to check data shapes, data types, and missing values.

### 2. Data Cleaning & Preprocessing
* **Handling Missing Values:** Cleaned whitespace and converted non-numeric strings in the `TotalCharges` column into numeric values, filling any resulting nulls appropriately.
* **Feature Removal:** Dropped irrelevant identifier columns like `customerID` that do not contribute to predictive patterns.
* **Target Encoding:** Mapped the target variable `Churn` (`Yes`/`No`) into binary format (`1`/`0`).
* **Categorical Encoding:** Converted categorical features into numerical representation using one-hot encoding (`pd.get_dummies`) to prepare data for machine learning algorithms.

### 3. Feature Engineering & Exploratory Data Analysis (EDA)
* Analyzed relationships between customer features and churn rates. Key insights discovered:
  * **Contract Type:** Month-to-month customers exhibit significantly higher churn rates compared to those on 1-year or 2-year contracts.
  * **Internet Service:** Fiber optic users showed higher churn compared to DSL users, highlighting potential service or pricing pain points.
  * **Tenure:** Newer customers (low tenure) are at a much higher risk of churning.

### 4. Machine Learning Model Development
* **Train-Test Split:** Split the dataset into training and testing sets (80/20 split) with a fixed random state for reproducibility.
* **Feature Scaling:** Standardized numerical features using `StandardScaler` to ensure features like `MonthlyCharges` and `tenure` contribute equally.
* **Model Training:** Trained a robust **Random Forest Classifier** to capture complex non-linear relationships and interactions between customer attributes.
* **Evaluation Metrics:** Evaluated model performance using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

### 5. Interactive Streamlit Dashboard (`churn_app.py`)
Built a web application featuring:
* **Real-Time Prediction Tab:** Allows users to modify customer profiles (Tenure, Monthly Charges, Contract type, Payment method, etc.) via sidebar sliders/dropdowns to instantly predict whether a customer is at risk of churning.
* **EDA & Insights Tab:** Displays key business metrics (Total Customers, Overall Churn Rate, Average Charges) alongside dynamic visual charts analyzing churn patterns by contract and internet service.

---

## 📁 Repository Structure
```text
├── churn_app.py                  # Main Streamlit application code
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset file
├── requirements.txt              # Required dependencies for deployment
└── README.md                     # Project documentation
