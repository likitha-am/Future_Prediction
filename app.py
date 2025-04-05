import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
import joblib

# Load model and preprocessing tools
model = joblib.load("career_model.pkl")
tfidf = joblib.load("tfidf.pkl")
scaler = joblib.load("scaler.pkl")
ohe_objects = joblib.load("ohe_objects.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load data for dropdown options
df = pd.read_csv("xceldoc.csv")
df.columns = df.columns.str.strip()

education_levels = sorted(df["Highest Education Level"].dropna().unique().tolist())
subjects_list = sorted(df["Preferred Subjects in Highschool/College"].dropna().unique().tolist())
work_envs = sorted(df["Preferred Work Environment"].dropna().unique().tolist())
tech_levels = sorted(df["Tech-Savviness"].dropna().unique().tolist())

# Streamlit UI
st.set_page_config(page_title="Career Predictor", page_icon="🎯")
st.title("🎓 Future Career Prediction App")
st.markdown("Please enter your details below to predict your future career!")

# Input fields
age = st.slider("Age", 10, 40, step=1)
education = st.selectbox("Highest Education Level", education_levels)
subjects = st.selectbox("Preferred Subjects in Highschool/College", subjects_list)
performance = st.slider("Academic Performance (CGPA or %)", 0.0, 10.0, step=0.1)
work_env = st.selectbox("Preferred Work Environment", work_envs)
risk = st.slider("Risk-Taking Ability (1 to 10)", 1, 10)
tech = st.selectbox("Tech-Savviness", tech_levels)
finance = st.slider("Financial Stability (1 to 10)", 1, 10)

# Predict button
if st.button("🔮 Predict My Career"):
    try:
        # Preprocess input
        X_num = scaler.transform([[age, performance, risk, finance]])
        X_text = tfidf.transform([subjects])
        cat_inputs = [education, work_env, tech]
        encoded = []
        for i, val in enumerate(cat_inputs):
            enc = ohe_objects[i].transform([[val]])
            encoded.append(enc)
        X_cat = np.hstack(encoded)
        final_input = hstack([csr_matrix(X_num), csr_matrix(X_cat), X_text])

        # Make prediction
        prediction = model.predict(final_input)
        career = label_encoder.inverse_transform(prediction)[0]

        st.success(f"🎯 Your predicted career is: **{career}**")
    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
