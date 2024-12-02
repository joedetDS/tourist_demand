import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib  # For loading the scaler
import matplotlib.pyplot as plt

# Load the pre-trained attention-enhanced model
model = tf.keras.models.load_model('lstm_attention_total_arrival.h5')

# Load the pre-saved scaler
scaler = joblib.load('scaler.pkl')

# Function to prepare the input data
def prepare_input(data, time_steps):
    data_scaled = scaler.transform(data)
    X = []
    for i in range(len(data_scaled) - time_steps):
        X.append(data_scaled[i:i+time_steps])
    return np.array(X)

# Streamlit app interface
st.title("Tourist Arrival Forecasting with Attention-Enhanced LSTM")

# Sidebar for file upload
st.sidebar.title("Upload Your CSV File")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

data = None  # Initialize the data variable

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", data.head(), data.shape)

    # Ensure the uploaded column is numeric and contains no NaN values
    if "Tourist_Arrival" not in data.columns:
        st.error("Column 'Tourist_Arrival' not found in the dataset. Please check your data.")
    else:
        # Check for missing values
        if data["Tourist_Arrival"].isnull().any():
            st.error("The column 'Tourist_Arrival' contains missing values. Please clean your data.")
        else:
            # Ensure all values are numeric
            try:
                data["Tourist_Arrival"] = pd.to_numeric(data["Tourist_Arrival"], errors="coerce")
                if data["Tourist_Arrival"].isnull().any():  # Re-check after coercion
                    st.error("Some values in 'Tourist_Arrival' are not numeric and were converted to NaN. Please fix your data.")
                else:
                    st.write("Data is valid and ready for processing.")
            except Exception as e:
                st.error(f"An error occurred while validating the data: {e}")

# Ensure that we only use a maximum of 50 rows
if data is not None:
    data = data.head(50)

# Fixed time steps (remove user selection)
time_steps = 10  # Set this to whatever fixed value you need, for example 10.

# Slider for forecast months
forecast_months = st.sidebar.number_input("Months to Forecast", min_value=1, value=3, step=1)

# Process data and make predictions
if st.sidebar.button("Generate Forecast"):
    if data is None:
        st.error("Please upload a CSV file.")
    elif len(data) < time_steps:
        st.error(f"The dataset must have at least {time_steps} rows to match the selected time window.")
    else:
        try:
            # Scale the input data using the loaded scaler
            dt_arr = np.array(data["Tourist_Arrival"]).reshape(-1, 1)  # Ensure it's a 2D array for scaling
            data_scaled = scaler.transform(dt_arr)
            data["Scaled"] = data_scaled

            # Prepare the input data
            input_data = prepare_input(data[["Scaled"]], time_steps)

            if len(input_data) == 0:
                st.error(f"Not enough data to create time steps. Ensure the dataset has at least {time_steps} rows.")
            else:
                # Extract the last data points (reshape to match model input)
                last_window = dt_arr[-time_steps:].reshape(1, time_steps, 1)

                forecast_values = []

                # Generate forecast for the requested number of months
                for month in range(forecast_months):
                    next_month_prediction = model.predict(last_window)

                    forecast_value = next_month_prediction[0][0]
                    forecast_values.append(forecast_value)

                    # Update the input window with the new prediction
                    last_window = np.append(last_window[:, 1:, :], next_month_prediction.reshape(1, 1, 1), axis=1)

                # Display results for each forecasted month
                for month, forecast_value in enumerate(forecast_values, 1):
                    st.write(f"Month {month} forecast: {forecast_value:.2f}")

                # Visualization
                st.write("Observed vs Predicted Time Series:")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(range(len(data)), data["Tourist_Arrival"], label="Observed", color="blue")

                # Plot the forecasted values
                ax.plot(range(len(data), len(data) + forecast_months), forecast_values, label="Forecasted", color="orange", marker='o')
                ax.set_xlabel("Time [Monthly]", fontsize=12)
                ax.set_ylabel("Tourist Arrival", fontsize=12)
                ax.set_title("Observed vs Predicted Tourist Arrivals", fontsize=14)
                ax.legend()
                st.pyplot(fig)

        except ValueError:
            st.error("Ensure the uploaded data matches the scaling requirements of the pre-trained model.")
