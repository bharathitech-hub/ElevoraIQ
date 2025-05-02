
import streamlit as st
import pandas as pd
import pickle
import os
import joblib

#  Load the trained model here
model = joblib.load("model.pkl")



# Set Streamlit page config
st.set_page_config(page_title="ElevoraIQ", layout="centered")
st.title("ElevoraIQ - Professional Aptitude & Compensation Intelligence Framework")

# Load the model using a safe relative path
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.header("Candidate Profile")
aptitude = st.slider("Aptitude Score (0 - 100)", 0, 100, 50)
domain_knowledge = st.slider("Domain Knowledge Score (0 - 100)", 0, 100, 50)
experience = st.number_input("Years of Experience", min_value=0.0, step=0.5)
current_comp = st.number_input("Current Compensation (₹)", min_value=0.0, step=10000.0)
education = st.selectbox("Highest Education Level", ["Bachelors", "Masters", "PhD"])

# Encode education
education_map = {"Bachelors": 0, "Masters": 1, "PhD": 2}
education_level = education_map[education]

# Input DataFrame
input_df = pd.DataFrame({
    "aptitude": [aptitude],
    "domain_knowledge": [domain_knowledge],
    "experience": [experience],
    "current_compensation": [current_comp],
    "education_level": [education_level]
})

# Predict
if st.button("Predict Expected Salary"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Expected Compensation: ₹{prediction:,.2f}")
