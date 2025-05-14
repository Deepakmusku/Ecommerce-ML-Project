# prophet_model.py

import os
import pandas as pd
from prophet import Prophet

def run_prophet():
    """Function to run Prophet model on cleaned sales data."""
    
    # Get the current working directory
    base_path = os.getcwd()
    
    # Construct the input file path
    input_file = os.path.join(base_path, "data", "cleaned_sales.csv")
    print(f"Input path: {input_file}")

    # Check if the file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # Load data
    try:
        data = pd.read_csv(input_file)
        print(f"Successfully loaded data from {input_file}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Display the initial data
    print("\nData Head:\n", data.head())
    print("\nData Info:\n")
    print(data.info())

    # Rename columns for Prophet
    data = data.rename(columns={"date": "ds", "sales": "y"})

    # Check if the required columns are present
    if not {"ds", "y"}.issubset(data.columns):
        print("Error: Required columns 'ds' and 'y' are not present in the data.")
        return

    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(data)

    # Create future dataframe and predict
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Display the forecast
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # Save the forecast as a CSV file
    output_file = os.path.join(base_path, "data", "sales_forecast.csv")
    forecast.to_csv(output_file, index=False)
    print(f"\nForecast successfully saved to {output_file}")

if __name__ == "__main__":
    run_prophet()
