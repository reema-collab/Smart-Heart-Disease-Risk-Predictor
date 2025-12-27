import streamlit as st
import numpy as np
import pickle

# Load model & scaler
model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Smart Health Predictor", page_icon="❤️")

st.title("❤️ Smart Heart Disease Risk Predictor")
st.write("Enter your health details below to predict your heart disease risk.")

# ----------- User Inputs -----------
age = st.slider("Age", 1, 120, 30)
sex = st.radio("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.number_input("Serum Cholesterol (mg/dl)", 80, 600, 200)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [1, 0])
restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", 50, 250, 150)
exang = st.radio("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
ca = st.selectbox("Major Vessels Colored by Fluoroscopy (0-4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (1-3)", [1, 2, 3])

# ----------- Prediction -----------
if st.button("Predict Risk"):
    # Convert input to numpy array
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])

    # Scale input
    scaled_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_data)[0]

    if prediction == 1:
        st.error("⚠ High Risk of Heart Disease! Please consult a doctor.")
    else:
        st.success("✅ Low Risk of Heart Disease. Keep maintaining a healthy lifestyle!")
