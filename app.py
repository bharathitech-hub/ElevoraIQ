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
        original_df (pd.DataFrame): The original training DataFrame for consistent encoding and mean imputation.
        scaler (StandardScaler): The fitted StandardScaler.

    Returns:
        np.ndarray: Preprocessed and scaled numerical data for prediction.
    """
    if original_df is None:
        st.error("Original training data is required for preprocessing.")
        return None

    input_df = pd.DataFrame([data])

    # --- Preprocessing steps (MUST MATCH the training notebook) ---
    for col in ['Gender', 'Work_Gap_Status']:
        le = LabelEncoder()
        if col in original_df.columns and col in input_df.columns:
            # Fit on the combined unique values from original and input to handle unseen values
            combined_values = pd.concat([original_df[col].astype(str), input_df[col].astype(str)]).unique()
            le.fit(combined_values)
            input_df[col] = le.transform(input_df[col].astype(str))
        elif col in input_df.columns:
            input_df[col] = input_df[col].astype(str).fillna('Unknown') # Handle missing in input
            # If the column wasn't in the original, we can't reliably encode.
            # Consider a default encoding or handling strategy.
            input_df[col] = 0 # Default encoding if not in original
        else:
            input_df[col] = 0 # If not provided in input

    input_df['Skills'] = input_df['Skills'].fillna('None')
    skills_dummies = input_df['Skills'].str.get_dummies(sep=';')
    input_df = pd.concat([input_df.drop('Skills', axis=1), skills_dummies], axis=1)
    input_df['Skills_Count'] = input_df['Skills_Count'] = input_df['Skills'].apply(lambda x: len(x.split(';')) if x != 'None' else 0)


    input_df['Certifications'] = input_df['Certifications'].fillna('None')
    certifications_dummies = input_df['Certifications'].str.get_dummies(sep=';')
    input_df = pd.concat([input_df.drop('Certifications', axis=1), certifications_dummies], axis=1)
    input_df['Certification_Count'] = input_df['Certification_Count'] = input_df['Certifications'].apply(lambda x: len(x.split(';')) if x != 'None' else 0)

    categorical_cols = ['Education_Level', 'Industry', 'Location', 'Job_Role', 'Company_Type', 'Soft_Skills', 'Notice_Period']
    input_df = pd.get_dummies(input_df, columns=categorical_cols, dummy_na=False)

    input_df = input_df.drop(columns=['Candidate_ID'], errors='ignore')

    # Fill missing columns with 0 (to match training data)
    if original_df is not None:
        missing_cols = set(original_df.drop(columns=['Expected_Salary']).columns) - set(input_df.columns)
        for c in missing_cols:
            input_df[c] = 0
        input_df = input_df[original_df.drop(columns=['Expected_Salary']).columns] # Ensure correct order

    # Fill remaining NaNs with the mean of the original training data
    if original_df is not None:
        input_df = input_df.fillna(original_df.drop(columns=['Expected_Salary']).mean())
    else:
        input_df = input_df.fillna(0) # If original_df couldn't be loaded

    if scaler is not None:
        try:
            input_df_scaled = scaler.transform(input_df)
            return input_df_scaled
        except Exception as e:
            st.error(f"Error during scaling: {e}")
            return None
    else:
        return input_df.values # Return unscaled if scaler is not available

def main():
    st.title("Salary Expectation Prediction")
    st.write("Enter the candidate details to predict their expected salary.")

    if model is None or original_df is None:
        st.warning("The model or training data could not be loaded. Please check the file paths.")
        return

    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    work_gap_status = st.selectbox("Work Gap Status", ["Yes", "No"])
    education_level = st.selectbox("Education Level", original_df['Education_Level'].unique() if original_df is not None else ["MCA", "B.Tech"])
    industry = st.selectbox("Industry", original_df['Industry'].unique() if original_df is not None else ["IT Services", "Product Based"])
    location = st.selectbox("Location", original_df['Location'].unique() if original_df is not None else ["Bangalore", "Hyderabad"])
    job_role = st.selectbox("Job Role", original_df['Job_Role'].unique() if original_df is not None else ["Software Developer", "Data Scientist"])
    company_type = st.selectbox("Company Type", original_df['Company_Type'].unique() if original_df is not None else ["MNC", "Startup"])
    soft_skills = st.selectbox("Soft Skills", original_df['Soft_Skills'].unique() if original_df is not None else ["Excellent", "Good"])
    notice_period = st.selectbox("Notice Period", original_df['Notice_Period'].unique() if original_df is not None else ["30 Days", "15 Days"])
    years_of_experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)
    aptitude_score = st.number_input("Aptitude Score", min_value=0, max_value=100, value=80)
    interview_score = st.number_input("Interview Score", min_value=0, max_value=100, value=75)
    current_salary = st.number_input("Current Salary", min_value=0, value=1000000)
    skills = st.text_input("Skills (separate by ';')", "Python;SQL")
    certifications = st.text_input("Certifications (separate by ';')", "AWS Certification")

    if st.button("Predict Expected Salary"):
        input_data = {
            "Gender": gender,
            "Work_Gap_Status": work_gap_status,
            "Education_Level": education_level,
            "Industry": industry,
            "Location": location,
            "Job_Role": job_role,
            "Company_Type": company_type,
            "Soft_Skills": soft_skills,
            "Notice_Period": notice_period,
            "Years_of_Experience": years_of_experience,
            "Aptitude_Score": aptitude_score,
            "Interview_Score": interview_score,
            "Current_Salary": current_salary,
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





