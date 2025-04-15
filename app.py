import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.set_page_config(page_title="ElevoraIQ", layout="centered")

st.title("Enter Candidate Details")

aptitude = st.slider("Aptitude Score (0 - 100)", 0, 100, 50)
domain_knowledge = st.slider("Domain Knowledge Score (0 - 100)", 0, 100, 50)
experience = st.number_input("Years of Experience", min_value=0.0, format="%.2f")
current_comp = st.number_input("Current Compensation (₹)", min_value=0.0, format="%.2f")

education_level = st.selectbox("Highest Education Level", [
    "Bachelors", "Masters", "PhD", "Diploma", "Other"
])

# Prepare input
input_df = pd.DataFrame({
    "aptitude": [aptitude],
    "domain_knowledge": [domain_knowledge],
    "experience": [experience],
    "current_compensation": [current_comp],
    "education_level": [education_level]
})

# Ensure input columns match the model's expectations
input_df = input_df[model.feature_names_in_]

# Predict salary
if st.button("Predict Expected Salary"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Expected Compensation: ₹{prediction:,.2f}")
