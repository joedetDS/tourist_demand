import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib  # For loading the scalers
import matplotlib.pyplot as plt

# Load the pre-trained LSTM-Attention model
model = tf.keras.models.load_model('lstm_attention_model.keras')

# Load the pre-saved scalers
scaler_features = joblib.load('scaler_features.pkl')
scaler_target = joblib.load('scaler_target.pkl')

# Function to prepare input data
def prepare_input(data, time_steps):
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
    return np.array(X)

# Streamlit app interface
st.title("Tourism Demand Forecasting with LSTM-Attention")

# Sidebar for user inputs
st.sidebar.title("User Input")

# File upload for data input
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# User input for forecast months
forecast_months = st.sidebar.number_input("Number of months to forecast", min_value=1, value=3, step=1)

# Main processing
if uploaded_file is not None:
    try:
        # Load data
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:", data.head())

        # Ensure required columns are present
        required_columns = ['Humidity', 'Rainfall', 'Events', 'Weather', 'Tourism_Demand']
        if not all(col in data.columns for col in required_columns):
            st.error(f"Missing required columns. Ensure the dataset contains: {', '.join(required_columns)}.")
        else:
            # Separate features and target
            features = ['Humidity', 'Rainfall', 'Events', 'Weather']
            target = 'Tourism_Demand'

            X = data[features].values
            y = data[[target]].values

            # Scale features and target
            X_scaled = scaler_features.transform(X)
            y_scaled = scaler_target.transform(y)

            # Prepare input data for time step = 12
            time_steps = 12
            X_prepared = prepare_input(X_scaled, time_steps)
            y_prepared = y_scaled[time_steps:]

            if len(X_prepared) == 0:
                st.error(f"Not enough data to create time steps. Ensure the dataset has at least {time_steps} rows.")
            else:
                # Forecast
                last_window = X_scaled[-time_steps:].reshape(1, time_steps, len(features))
                forecast_values = []

                for _ in range(forecast_months):
                    next_prediction = model.predict(last_window)
                    forecast_values.append(next_prediction[0, 0])

                    # Update the last window
                    next_feature = np.append(last_window[:, 1:, :], next_prediction.reshape(1, 1, -1), axis=1)
                    last_window = next_feature

                # Inverse transform the forecasted values
                forecast_values = scaler_target.inverse_transform(np.array(forecast_values).reshape(-1, 1)).flatten()

                # Display results
                for month, value in enumerate(forecast_values, start=1):
                    st.write(f"Forecast for month {month}: {value:.2f}")

                # Visualization
                observed = data[target].values[-(len(y_prepared)):] if len(y_prepared) > 0 else []

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(range(len(observed)), observed, label="Observed", color="blue")
                ax.plot(range(len(observed), len(observed) + forecast_months), forecast_values, label="Forecasted", color="orange", marker='o')
                ax.set_xlabel("Time [Months]")
                ax.set_ylabel("Tourism Demand")
                ax.set_title("Observed vs Predicted Tourism Demand")
                ax.legend()
                st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")
