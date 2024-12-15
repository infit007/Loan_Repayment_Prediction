import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('model/loan_repayment_model.pkl')

# Streamlit app UI
st.title('Loan Repayment Prediction App')

st.markdown("""
This app predicts the likelihood of loan repayment and provides actionable insights based on the input features.
""")

# Input features (same as the ones used during training)
purpose = st.selectbox("Purpose of Loan", options=['debt_consolidation', 'credit_card', 'home_improvement', 'small_business'])
credit_policy = st.selectbox("Credit Policy", options=[1, 0])  # Assuming 1=Approved, 0=Denied
int_rate = st.number_input("Interest Rate (e.g., 0.1 for 10%)", min_value=0.0, max_value=1.0, value=0.1)
installment = st.number_input("Installment Amount", min_value=0, value=1000)
log_annual_inc = st.number_input("Log Annual Income (e.g., 11 for ~60k/year)", min_value=0.0, value=11.0)
dti = st.number_input("Debt-to-Income Ratio (DTI)", min_value=0.0, value=20.0)
fico = st.number_input("FICO Score", min_value=300, max_value=850, value=700)
days_with_cr_line = st.number_input("Days with Credit Line", min_value=0, value=3650)
revol_bal = st.number_input("Revolving Balance", min_value=0, value=5000)
revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, max_value=100.0, value=30.0)
inq_last_6mths = st.number_input("Inquiries in Last 6 Months", min_value=0, value=1)
delinq_2yrs = st.number_input("Delinquencies in Last 2 Years", min_value=0, value=0)
pub_rec = st.number_input("Public Records", min_value=0, value=0)

# Encoding categorical data (e.g., 'purpose')
purpose_map = {'debt_consolidation': 0, 'credit_card': 1, 'home_improvement': 2, 'small_business': 3}
purpose_encoded = purpose_map.get(purpose, -1)

# Prepare the input data for prediction
input_data = np.array([
    purpose_encoded, credit_policy, int_rate, installment, log_annual_inc, dti,
    fico, days_with_cr_line, revol_bal, revol_util, inq_last_6mths,
    delinq_2yrs, pub_rec
]).reshape(1, -1)

# Prediction
if st.button('Predict'):
    # Predict the outcome
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0]

    # Display prediction and probability
    if prediction == 0:
        st.success("Prediction: Loan will be fully repaid.")
        st.write(f"Confidence: {probability[0] * 100:.2f}%")
    else:
        st.error("Prediction: Loan will not be repaid.")
        st.write(f"Confidence: {probability[1] * 100:.2f}%")

    # Insights based on features
    st.markdown("### Actionable Insights:")
    if dti > 35:
        st.write("- **High Debt-to-Income Ratio (DTI):** Consider reducing your DTI to improve repayment likelihood.")
    if fico < 650:
        st.write("- **Low FICO Score:** Improving your credit score could significantly enhance your repayment potential.")
    if revol_util > 50:
        st.write("- **High Revolving Utilization:** Aim to keep utilization below 30% to demonstrate better credit management.")
    if int_rate > 0.15:
        st.write("- **High Interest Rate:** Look for loans with lower interest rates to reduce repayment burden.")

    st.markdown("### Feature Details:")
    st.write(f"- **Purpose:** {purpose}")
    st.write(f"- **FICO Score:** {fico}")
    st.write(f"- **DTI:** {dti}")
    st.write(f"- **Interest Rate:** {int_rate}")
