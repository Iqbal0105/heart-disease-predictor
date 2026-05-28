import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/heart_disease_model.pkl")

st.set_page_config(page_title="Prediksi Penyakit Jantung", layout="centered")

st.title("🫀 Prediksi Risiko Penyakit Jantung")
st.markdown("Masukkan data berikut untuk memprediksi kemungkinan terkena penyakit jantung:")

# Input fitur
age = st.selectbox("Kategori Usia", [
    "18-24", "25-29", "30-34", "35-39", "40-44",
    "45-49", "50-54", "55-59", "60-64", "65-69",
    "70-74", "75-79", "80 atau lebih"
])
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, step=0.1)
gen_health = st.selectbox("Kesehatan Umum (General Health)", ["Sangat Baik", "Baik", "Biasa", "Buruk", "Sangat Buruk"])
smoking = st.selectbox("Apakah Merokok?", ["Yes", "No"])
diabetic = st.selectbox("Status Diabetes", ["Yes", "No", "Yes (during pregnancy)", "No, borderline diabetes"])
physical_activity = st.selectbox("Aktif Secara Fisik?", ["Yes", "No"])
stroke = st.selectbox("Pernah Stroke?", ["Yes", "No"])
diff_walking = st.selectbox("Kesulitan Berjalan?", ["Yes", "No"])
kidney_disease = st.selectbox("Memiliki Penyakit Ginjal?", ["Yes", "No"])
sex = st.selectbox("Jenis Kelamin", ["Male", "Female"])

# Prediksi
if st.button("🔍 Prediksi"):
    input_df = pd.DataFrame([{
        "AgeCategory": age,
        "BMI": bmi,
        "GenHealth": gen_health,
        "Smoking": smoking,
        "Diabetic": diabetic,
        "PhysicalActivity": physical_activity,
        "Stroke": stroke,
        "DiffWalking": diff_walking,
        "KidneyDisease": kidney_disease,
        "Sex": sex
    }])
    
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Individu ini berisiko tinggi terkena penyakit jantung (Probabilitas: {prob:.2f})")
    else:
        st.success(f"✅ Individu ini berisiko rendah terkena penyakit jantung (Probabilitas: {prob:.2f})")
