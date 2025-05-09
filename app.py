import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# Load the trained model
MODEL_PATH = 'salary_model.pkl'
if os.path.exists(MODEL_PATH):
    try:
        model = pickle.load(open(MODEL_PATH, 'rb'))
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        model = None
else:
    st.error(f"Model file not found at: {MODEL_PATH}. Please ensure 'salary_model.pkl' is in the same directory.")
    model = None

# Load the original training data (for consistent preprocessing)
DATA_PATH = 'Salary_Data.csv'
if os.path.exists(DATA_PATH):
    try:
        original_df = pd.read_csv(DATA_PATH)
    except Exception as e:
        st.error(f"Error loading the training data: {e}")
        original_df = None
else:
    st.error(f"Training data file not found at: {DATA_PATH}. Please ensure 'Salary_Data.csv' is in the same directory.")
    original_df = None

# Load the scaler (assuming it was saved)
SCALER_PATH = 'scaler.pkl'
if os.path.exists(SCALER_PATH):
    try:
        scaler = pickle.load(open(SCALER_PATH, 'rb'))
    except Exception as e:
        st.error(f"Error loading the scaler: {e}")
        scaler = None
else:
    st.warning("Scaler file 'scaler.pkl' not found. Predictions might be less accurate.")
    scaler = None


def preprocess_input(data, original_df, scaler):
    """
    Preprocesses the input data to match the training data format.

    Args:
        data (dict): Input data as a dictionary.
        original_df (pd.DataFrame): The original training DataFrame.
        scaler (StandardScaler): The fitted StandardScaler.

    Returns:
        np.ndarray: Preprocessed and scaled numerical data for prediction.
    """

    if original_df is None:
        st.error("Original training data is required for preprocessing.")
        return None

    input_df = pd.DataFrame([data])

    # --- Preprocessing steps (MUST MATCH the training notebook) ---
    # Rename input columns to match training data (CRITICAL FIX)
    input_df = input_df.rename(columns={
        'aptitude': 'Aptitude_Score',
        'current_compensation': 'Current_Salary',
        'domain_knowledge': 'Domain_Knowledge',  # If it exists; otherwise handle KeyError
        'education_level': 'Education_Level',
        'experience': 'Years_of_Experience',
        # ... add any other renames needed based on the error log ...
    }, errors='ignore')  # 'ignore' prevents errors if a column isn't present

    for col in ['Gender', 'Work_Gap_Status']:
        le = LabelEncoder()
        if col in original_df.columns and col in input_df.columns:
            combined_values = pd.concat([original_df[col].astype(str), input_df[col].astype(str)]).unique()
            le.fit(combined_values)
            input_df[col] = le.transform(input_df[col].astype(str))
        elif col in input_df.columns:
            input_df[col] = input_df[col].astype(str).fillna('Unknown')
            input_df[col] = 0  # Or a more suitable default
        else:
            input_df[col] = 0

    input_df['Skills'] = input_df['Skills'].fillna('None')
    skills_dummies = input_df['Skills'].str.get_dummies(sep=';', prefix='Skills')  # Add prefix
    input_df = pd.concat([input_df.drop('Skills', axis=1, errors='ignore'), skills_dummies], axis=1,  ignore_index=False)
    input_df['Skills_Count'] = input_df['Skills'].apply(lambda x: len(x.split(';')) if x != 'None' else 0)

    input_df['Certifications'] = input_df['Certifications'].fillna('None')
    certifications_dummies = input_df['Certifications'].str.get_dummies(sep=';', prefix='Certifications')  # Add Prefix
    input_df = pd.concat([input_df.drop('Certifications', axis=1, errors='ignore'), certifications_dummies], axis=1,  ignore_index=False)
    input_df['Certification_Count'] = input_df['Certifications'].apply(lambda x: len(x.split(';')) if x != 'None' else 0)

    categorical_cols = ['Education_Level', 'Industry', 'Location', 'Job_Role', 'Company_Type', 'Soft_Skills', 'Notice_Period']
    input_df = pd.get_dummies(input_df, columns=categorical_cols, dummy_na=False)

    input_df = input_df.drop(columns=['Candidate_ID'], errors='ignore')

    if original_df is not None:
        missing_cols = set(original_df.drop(columns=['Expected_Salary'], errors='ignore').columns) - set(input_df.columns)
        for c in missing_cols:
            input_df[c] = 0
        input_df = input_df[original_df.drop(columns=['Expected_Salary'], errors='ignore').columns]  # Ensure correct order
    else:
        input_df = input_df.fillna(0)

    if original_df is not None:
        input_df = input_df.fillna(original_df.drop(columns=['Expected_Salary'], errors='ignore').mean())
    else:
        input_df = input_df.fillna(0)

    if scaler is not None:
        try:
            input_df_scaled = scaler.transform(input_df)
            return input_df_scaled
        except Exception as e:
            st.error(f"Error during scaling: {e}")
            return None
    else:
        return input_df.values


def main():
    st.title("Salary Expectation Prediction")
    st.write("Enter the candidate details to predict their expected salary.")

    if model is None or original_df is None:
        st.warning("The model or training data could not be loaded. Please check the file paths.")
        return

    # Create input fields (make sure names match what the user will provide)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="gender")
    work_gap_status = st.selectbox("Work Gap Status", ["Yes", "No"], key="work_gap")
    education_level = st.selectbox("Education Level", original_df['Education_Level'].unique() if original_df is not None else ["MCA", "B.Tech"], key="education")
    industry = st.selectbox("Industry", original_df['Industry'].unique() if original_df is not None else ["IT Services", "Product Based"], key="industry")
    location = st.selectbox("Location", original_df['Location'].unique() if original_df is not None else ["Bangalore", "Hyderabad"], key="location")
    job_role = st.selectbox("Job Role", original_df['Job_Role'].unique() if original_df is not None else ["Software Developer", "Data Scientist"], key="job_role")
    company_type = st.selectbox("Company Type", original_df['Company_Type'].unique() if original_df is not None else ["MNC", "Startup"], key="company")
    soft_skills = st.selectbox("Soft Skills", original_df['Soft_Skills'].unique() if original_df is not None else ["Excellent", "Good"], key="soft_skills")
    notice_period = st.selectbox("Notice Period", original_df['Notice_Period'].unique() if original_df is not None else ["30 Days", "15 Days"], key="notice")
    experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5, key="experience")  # Changed variable name
    aptitude = st.number_input("Aptitude Score", min_value=0, max_value=100, value=80, key="aptitude")  # Changed variable name
    interview_score = st.number_input("Interview Score", min_value=0, max_value=100, value=75, key="interview")
    current_compensation = st.number_input("Current Salary", min_value=0, value=1000000, key="current_salary")  # Changed variable name
    skills = st.text_input("Skills (separate by ';')", "Python;SQL", key="skills")
    certifications = st.text_input("Certifications (separate by ';')", "AWS Certification", key="certifications")

    if st.button("Predict Expected Salary"):
        input_data = {
            "Gender": gender,
            "Work_Gap_Status": work_gap_status,
            "education_level": education_level,  # Changed key
            "industry": industry,  # Changed key
            "location": location,  # Changed key
            "job_role": job_role,  # Changed key
            "company_type": company_type,  # Changed key
            "soft_skills": soft_skills,  # Changed key
            "notice_period": notice_period,  # Changed key
            "experience": years_of_experience,  # Changed key
            "aptitude": aptitude_score,  # Changed key
            "Interview_Score": interview_score,
            "current_compensation": current_salary,  # Changed key
            "Skills": skills,
            "Certifications": certifications
        }

        processed_input = preprocess_input(input_data, original_df, scaler)

        if processed_input is not None and model is not None:
            try:
                prediction = model.predict(processed_input)[0]
                st.success(f"Predicted Expected Salary: ₹{int(np.round(prediction))}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")


if __name__ == '__main__':
    main()





