import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from datetime import timedelta


#Load Model dan Scaler 
@st.cache_resource
def load_model():
    with open("gold_price_prediction.tflite", "rb") as f:
        model = f.read()
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()
    return interpreter

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

interpreter = load_model()
scaler = load_scaler()

# Fungsi Prediksi 
def predict_lstm_sequence(input_sequence, steps=60):
    forecast = []
    current_input = input_sequence.reshape(1, input_sequence.shape[0], 1)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for _ in range(steps):
        interpreter.set_tensor(input_details[0]['index'], current_input.astype(np.float32))
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])[0][0]
        forecast.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    return forecast

# UI 
st.title("Prediksi Harga Emas 60 Hari ke Depan")
st.markdown("Aplikasi ini menggunakan model LSTM untuk memprediksi harga penutupan emas berdasarkan data historis.")

df = pd.read_csv("FINAL_USO.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

st.subheader("Visualisasi Harga Historis")
st.line_chart(df.set_index('Date')['Close'])

# Forecasting 
st.subheader("Prediksi Harga 60 Hari ke Depan")
window_size = 60
data = df[['Close']]
data_scaled = scaler.transform(data)

# Ambil 60 data terakhir untuk prediksi
last_sequence = data_scaled[-window_size:]
forecast = predict_lstm_sequence(last_sequence, steps=60)

# Tampilkan hasil prediksi
last_date = df['Date'].iloc[-1]
future_dates = [last_date + timedelta(days=i+1) for i in range(60)]

forecast_df = pd.DataFrame({'Tanggal': future_dates, 'Prediksi Harga': forecast.flatten()})
forecast_df.set_index('Tanggal', inplace=True)

st.line_chart(forecast_df)

# Tabel Data
with st.expander("Lihat Data Prediksi"):
    st.dataframe(forecast_df.style.format({'Prediksi Harga': '{:,.2f}'}))

# Unduh Hasil Prediksi
csv = forecast_df.to_csv().encode('utf-8')
st.download_button("Unduh Hasil Prediksi (CSV)", data=csv, file_name='prediksi_emas_60_hari.csv', mime='text/csv')
