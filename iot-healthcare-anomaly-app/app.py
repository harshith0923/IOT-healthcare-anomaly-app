import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from models.lstm_autoencoder import LSTMAutoencoder
from utils.preprocessing import preprocess_data
import torch

st.title("🩺 IoT Healthcare Anomaly Detection (LSTM)")
st.markdown("Upload patient health data (CSV) to detect anomalies.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data")
    st.write(df.head())

    data_tensor = preprocess_data(df)

    model = LSTMAutoencoder(input_dim=data_tensor.shape[2])
    model.load_state_dict(torch.load("models/lstm_model.pt", map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        reconstructed = model(data_tensor)
        loss_fn = torch.nn.MSELoss(reduction='none')
        losses = loss_fn(reconstructed, data_tensor).mean(dim=(1, 2)).numpy()

    threshold = losses.mean() + 2 * losses.std()
    df['Anomaly Score'] = losses
    df['Anomaly'] = df['Anomaly Score'] > threshold

    st.subheader("📉 Anomaly Detection Results")
    st.dataframe(df[['Anomaly Score', 'Anomaly']])

    fig, ax = plt.subplots()
    ax.plot(df.index, df['Anomaly Score'], label='Score')
    ax.axhline(threshold, color='r', linestyle='--', label='Threshold')
    ax.set_title("Anomaly Scores Over Time")
    ax.legend()
    st.pyplot(fig)

    st.download_button("Download Results as CSV", df.to_csv(index=False), file_name="anomaly_results.csv")

    st.success("✅ Anomaly detection complete.")

st.markdown("---")
st.markdown("_Future update: real-time sensor input support._")
