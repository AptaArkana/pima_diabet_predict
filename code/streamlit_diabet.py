import joblib as job
import streamlit as st
import numpy as np

# Load model
model_path = os.path.join('.', 'models', 'pima_diabet_predict')  # Atur jalur file dengan benar
diabet_model = job.load(model_path)

scaler_path = os.path.join('.', 'models', 'scaler')  # Atur jalur file dengan benar
sc = job.load(scaler_path)

# Title web
st.title('Sistem Prediksi Diabetes')

# Kolom
col1, col2 = st.columns(2)

# Buat form input fitur
with col1:
    Pregnancies = st.text_input('Input nilai Pregnancies')

with col2:
    Glucose = st.text_input('Input nilai Glucose')

with col1:
    BloodPressure = st.text_input('Input nilai BloodPressure')

with col2:
    BMI = st.text_input('Input nilai BMI')

with col1:
    DiabetesPedigreeFunction = st.text_input('Input nilai DiabetesPedigreeFunction')

with col2:
    Age = st.text_input('Input nilai Age')

# Code untuk prediksi
diabet_diagnosis = ''

# Button prediksi
if st.button('Test Prediksi Diabetes'):
    # Create an input array from user inputs
    input_data = np.array([Pregnancies, Glucose, BloodPressure, BMI, DiabetesPedigreeFunction, Age])

    # Reshape the input data to the shape expected by the scaler (1 sample with 6 features)
    input_data = input_data.reshape(1, -1)

    # Scale the input data
    scaled_input_data = sc.transform(input_data)
    
    diabet_predict = diabet_model.predict(scaled_input_data)

    if round(diabet_predict[0][0]) < 1:
        diabet_diagnosis = 'Pasien Tidak Terkena Diabetes'
    else:
        diabet_diagnosis = 'Pasien Terkena Diabetes'

    st.success(diabet_diagnosis)
