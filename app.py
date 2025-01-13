import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# App title
st.title("Tourism Demand Forecasting")

# Cache model and scalers for efficiency
@st.cache_resource
def load_model_and_scalers():
    model = load_model("lstm_attention_model.keras")
    scaler_features = joblib.load("scaler_features.pkl")
    scaler_target = joblib.load("scaler_target.pkl")
    return model, scaler_features, scaler_target

# Load the pre-trained model and scalers
loaded_model, scaler_features, scaler_target = load_model_and_scalers()

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Check if a file is uploaded
if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    tourism_data = pd.read_csv(uploaded_file)

    # Display the first five rows of the loaded dataset
    st.write("First five rows of the uploaded dataset:")
    st.dataframe(tourism_data.head())
    
    # Perform ordinal encoding on categorical columns
    ordinal_encoder = OrdinalEncoder()
    tourism_data[['Events', 'Weather']] = ordinal_encoder.fit_transform(
        tourism_data[['Events', 'Weather']]
    )
    
    # Select features and target
    features = ['Humidity', 'Rainfall', 'Events', 'Weather']
    target = 'Tourism_Demand'

    # Scale features and target
    scaled_features = scaler_features.transform(tourism_data[features])
    scaled_target = scaler_target.transform(tourism_data[[target]])
    
    # Prepare data for forecasting (time step = 12)
    time_step = 12
    if len(scaled_features) < time_step:
        st.error("Not enough data points to make a forecast. At least 12 rows are required.")
    else:
        # Use the last 12 steps for forecasting
        recent_data = scaled_features[-time_step:, :]

        # User input for how many months to forecast using a slider
        steps_ahead = st.slider("Select the number of months to forecast:", min_value=1, max_value=10, value=3)

        # Forecast function
        def forecast_future_values(model, recent_data, scaler, steps_ahead=3):
            predictions = []
            current_input = recent_data

            for _ in range(steps_ahead):
                # Predict the next value
                next_value_scaled = model.predict(current_input[np.newaxis, :, :])[0, 0]
                predictions.append(next_value_scaled)

                # Update the input: Remove the first time step and add the new prediction
                next_input = np.append(current_input[1:], [[*current_input[-1, :-1], next_value_scaled]], axis=0)
                current_input = next_input

            # Transform predictions back to original scale
            predictions_original_scale = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
            return predictions_original_scale

        # Generate forecast
        future_predictions = forecast_future_values(loaded_model, recent_data, scaler_target, steps_ahead=steps_ahead)

        # Round the predictions to the nearest integer
        future_predictions_int = np.round(future_predictions).astype(int)

        # Display the forecasted values as integers
        st.write(f"Forecasted Tourism Demand for the next {steps_ahead} months:")
        forecast_df = pd.DataFrame({
            "Month": [f"Month {i+1}" for i in range(steps_ahead)],
            "Forecasted Demand": future_predictions_int
        })
        st.dataframe(forecast_df)

        # Create a line plot for "Dataset vs Forecast"
        st.write("Dataset vs Forecast:")
        plt.figure(figsize=(10, 4))

        # Plot the entire dataset as a line
        plt.plot(
            range(1, len(tourism_data) + 1), 
            tourism_data[target], 
            label="Dataset (Observed)", 
            marker="o", 
            color="blue"
        )

        # Plot forecast starting from the last data point
        forecast_start = len(tourism_data)
        plt.plot(
            range(forecast_start + 1, forecast_start + steps_ahead + 1), 
            future_predictions_int, 
            label="Forecast", 
            marker="x", 
            color="red"
        )

        # Add vertical line to indicate forecast start
        plt.axvline(x=forecast_start, color="gray", linestyle="--", label="Forecast Start")

        # Set plot labels and title
        plt.title("Dataset vs Forecast")
        plt.xlabel("Time (Months)")
        plt.ylabel("Tourism Demand")
        plt.legend()
        plt.grid(True)
        
        # Show the plot in Streamlit
        st.pyplot(plt)

else:
    st.info("Please upload a CSV file to proceed.")
