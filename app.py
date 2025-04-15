import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('model.pkl')

# App title
st.title("Elevora - Professional Aptitude & Compensation Estimator")
st.subheader("Smart compensation predictions based on your skillset")

# User Inputs
st.header("Candidate Profile Input")
aptitude = st.slider("Aptitude Score (0-100)", 0, 100, 50)
domain_knowledge = st.slider("Domain Knowledge (0-100)", 0, 100, 50)
experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=1.0)
current_comp = st.number_input("Current Compensation (₹)", min_value=0.0, value=300000.0)
education = st.selectbox("Education Level", ["Bachelors", "Masters", "PhD"])

# Encode education
edu_level = {"Bachelors": 0, "Masters": 1, "PhD": 2}
education_level = edu_level[education]

# Prepare data (REMOVED skill_alignment to match trained model)
input_df = pd.DataFrame({
    "aptitude": [aptitude],
    "domain_knowledge": [domain_knowledge],
    "experience": [experience],
    "current_compensation": [current_comp],
    "education_level": [education_level]
})

# Predict
if st.button("Predict Salary"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Expected Salary: ₹{prediction:,.2f}")

# Optional Feature Ideas (add later)
st.markdown("---")
st.markdown("Want to improve this app?")
st.markdown("- Add PDF Export")
st.markdown("- Add Profile Comparison")
st.markdown("- Add Logging/Tracking Features")
