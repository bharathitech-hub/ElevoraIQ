import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('model.pkl')

# App title
st.set_page_config(page_title="ElevoraIQ", layout="centered")
st.title("ElevoraIQ - Aptitude & Compensation Estimator")
st.subheader("Predict your expected salary based on your skills and profile.")

# User Inputs
st.header("Enter Candidate Details")

aptitude = st.slider("Aptitude Score (0 - 100)", 0, 100, 50)
domain_knowledge = st.slider("Domain Knowledge Score (0 - 100)", 0, 100, 50)
experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, step=0.5)
current_comp = st.number_input("Current Compensation (₹)", min_value=0.0, step=10000.0)
education = st.selectbox("Highest Education Level", ["Bachelors", "Masters", "PhD"])

# Encode education level
education_map = {"Bachelors": 0, "Masters": 1, "PhD": 2}
education_level = education_map[education]

# Prepare input dataframe
input_df = pd.DataFrame({
    "aptitude": [aptitude],
    "domain_knowledge": [domain_knowledge],
    "experience": [experience],
    "current_compensation": [current_comp],
    "education_level": [education_level]
})

# Prediction
if st.button("Predict Expected Salary"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Expected Compensation: ₹{prediction:,.2f}")
