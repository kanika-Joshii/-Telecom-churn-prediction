import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- App Title & Description ---
st.title("📊 Telco Customer Churn Live Dashboard")
st.markdown("This interactive dashboard predicts customer churn and explores key risk factors in real-time.")

# --- Load Data & Model Pipeline ---
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = df.drop_duplicates()
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df

df = load_data()

@st.cache_resource
def train_model(data):
    # Preprocessing for baseline model training
    df_model = data.copy()
    df_model = pd.get_dummies(df_model, drop_first=True)
    
    X = df_model.drop(columns=["Churn"])
    y = df_model["Churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X.columns

model, scaler, feature_columns = train_model(df)

# --- Sidebar: User Inputs for Prediction ---
st.sidebar.header("🔍 Customer Profile Input")

def user_input_features():
    tenure = st.sidebar.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=70.0)
    total_charges = tenure * monthly_charges  # Approximation or dynamic input
    
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment_method = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    
    data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'InternetService': internet_service,
        'PaymentMethod': payment_method,
        'PaperlessBilling': paperless_billing,
        'TechSupport': tech_support,
        # Fill remaining columns with defaults to match training shape
        'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
        'PhoneService': 'Yes', 'MultipleLines': 'No', 'OnlineSecurity': 'No',
        'OnlineBackup': 'No', 'DeviceProtection': 'No', 'StreamingTV': 'No',
        'StreamingMovies': 'No'
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# --- Main Panel Layout ---
tab1, tab2 = st.tabs(["🚀 Churn Prediction", "📈 Exploratory Data Analysis"])

with tab1:
    st.subheader("Real-Time Churn Prediction")
    
    # Combine with main df to align dummy variables correctly
    combined_df = pd.concat([df.drop(columns=["Churn"]), input_df], axis=0)
    encoded_df = pd.get_dummies(combined_df, drop_first=True)
    
    # Extract the input row (the last row)
    X_input = encoded_df.tail(1)
    
    # Align columns with training feature columns
    X_input = X_input.reindex(columns=feature_columns, fill_value=0)
    X_input_scaled = scaler.transform(X_input)
    
    prediction = model.predict(X_input_scaled)
    prediction_proba = model.predict_proba(X_input_scaled)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Prediction Result")
        if prediction[0] == 1:
            st.error("⚠️ **High Risk:** This customer is likely to **CHURN**.")
        else:
            st.success("✅ **Low Risk:** This customer is likely to **STAY**.")
            
    with col2:
        st.markdown("### Churn Probability")
        st.metric(label="Probability of Churn", value=f"{prediction_proba[0][1]*100:.2f}%")

with tab2:
    st.subheader("Dataset Overview & Insights")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Overall Churn Rate", f"{df['Churn'].mean() * 100:.2f}%")
    col3.metric("Avg Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")
    
    st.markdown("---")
    
    # Visualizations
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### Churn by Contract Type")
        fig, ax = plt.subplots(figsize=(6, 4))
        contract_churn = df.groupby("Contract")["Churn"].mean() * 100
        contract_churn.plot(kind="bar", ax=ax, color="skyblue")
        ax.set_ylabel("Churn Rate (%)")
        st.pyplot(fig)
        
    with c2:
        st.markdown("#### Churn by Internet Service")
        fig, ax = plt.subplots(figsize=(6, 4))
        internet_churn = df.groupby("InternetService")["Churn"].mean() * 100
        internet_churn.plot(kind="bar", ax=ax, color="salmon")
        ax.set_ylabel("Churn Rate (%)")
        st.pyplot(fig)