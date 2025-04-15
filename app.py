import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")

# Set page configuration
st.set_page_config(page_title="ElevoraIQ", layout="centered")

# App title
st.title("Enter Candidate Details")

# Input fields
aptitude_score = st.slider("Aptitude Score (0 - 100)", 0, 100, 50)
domain_score = st.slider("Domain Knowledge Score (0 - 100)", 0, 100, 50)
experience = st.number_input("Years of Experience", min_value=0.0, step=0.5, format="%.2f")
current_comp = st.number_input("Current Compensation (₹)", min_value=0.0, step=500.0, format="%.2f")
education = st.selectbox("Highest Education Level", ["Bachelors", "Masters", "PhD"])

# Encode education level
education_map = {"Bachelors": 0, "Masters": 1, "PhD": 2}
education_encoded = education_map[education]

# Prediction trigger
if st.button("Predict Expected Salary"):
    # Create input DataFrame
    input_df = pd.DataFrame({
        "aptitude_score": [aptitude_score],
        "domain_score": [domain_score],
        "experience": [experience],
        "current_comp": [current_comp],
        "education_level": [education_encoded]
    })

    # Perform prediction
    prediction = model.predict(input_df)[0]

    # Display result
    st.success(f"Predicted Expected Compensation: ₹{prediction:,.2f}")
