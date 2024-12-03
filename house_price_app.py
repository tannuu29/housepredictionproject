import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle


# Load or create the model
def train_model():
    # Example dataset with house features (Size, Bedrooms, Age, Price)
    data = {
        'Size': [1500, 1800, 2500, 1200, 2200],
        'Bedrooms': [3, 4, 4, 2, 3],
        'Age': [10, 15, 5, 20, 8],
        'Price': [400000, 500000, 650000, 300000, 550000]
    }

    # Convert the dictionary into a DataFrame
    df = pd.DataFrame(data)

    # Prepare the features and target variable
    X = df[['Size', 'Bedrooms', 'Age']]  # Features
    y = df['Price']  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the trained model using pickle
    with open('house_price_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    return model


# Load the trained model from pickle file
def load_model():
    try:
        with open('house_price_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        return model
    except FileNotFoundError:
        # If the model doesn't exist, we train it
        return train_model()


# Streamlit UI
st.title("House Price Prediction App")

# Load the model (either from the pickle file or train it if not found)
model = load_model()

# Input fields for the user to enter house details
st.subheader("Enter the details of the house")

size = st.number_input("Size (in square feet)", min_value=100, max_value=10000, value=1500)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
age = st.number_input("Age of the house (in years)", min_value=0, max_value=100, value=10)

# When the user clicks the "Predict" button
if st.button("Predict"):
    # Prepare the input data for prediction
    input_data = np.array([[size, bedrooms, age]])

    # Predict the house price using the trained model
    predicted_price_usd = model.predict(input_data)[0]

    # Convert the predicted price to Indian Rupees (INR)
    conversion_rate = 83  # 1 USD = 83 INR (update if needed)
    predicted_price_inr = predicted_price_usd * conversion_rate

    # Display the predicted house price in both USD and INR
    st.write(f"The predicted house price is: ${predicted_price_usd:,.2f} USD")
    st.write(f"The predicted house price is: ₹{predicted_price_inr:,.2f} INR")
