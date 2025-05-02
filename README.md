# Elevora Professional Aptitude & Compensation Intelligence Framework

## Overview
**Elevora** is an intelligent prediction framework designed to estimate a candidate’s expected compensation by analyzing key professional attributes. It leverages machine learning to bridge the gap between aptitude, domain knowledge, current profile, and compensation trends, making it a valuable tool for professionals, recruiters, and HR teams.

## Features
- Predicts expected salary based on real-world professional factors
- Simple and interactive Streamlit interface
- Supports dynamic user input: aptitude, domain knowledge, skills, education level, etc.
- Clean visualization and actionable prediction output
- Model performance metrics included (MAE, RMSE, R² Score)
- Extendable design: export results as PDF, future logging & profile comparison integration

## How It Works
1. The user enters profile-related information.
2. The model (trained and saved as `model.pkl`) processes input through a regression algorithm.
3. Output is shown directly on the app with predicted compensation in ₹.

## Tech Stack
- Python
- Streamlit
- Pandas, NumPy
- scikit-learn
- joblib

## Project Files
- `app.py` – Streamlit frontend
- `model.pkl` – Trained ML model
- `requirements.txt` – Dependencies
- `Elevora_Professional_Aptitude_&_Compensation_Intelligence_Framework.ipynb` – Core notebook with EDA and model training

## How to Run
1. Clone the repo:
   ```
   git clone https://github.com/your-username/elevora-compensation-framework.git
   cd elevora-compensation-framework
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   streamlit run app.py
   ```

## Deployment
The application can be deployed seamlessly using [Streamlit Cloud](https://streamlit.io/cloud). Simply upload the repository, ensure `app.py` and `requirements.txt` are in the root folder, and click “Deploy”.

## Author
Driven by vision and fueled by purpose, I build solutions that matter.  
Every line of code reflects relentless dedication and a desire to create real-world impact.  
This project is more than tech — it's a personal mission to grow, contribute, and lead.

## License
This project is licensed under the [MIT License](LICENSE).