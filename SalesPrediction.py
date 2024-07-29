import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("Advertising.csv")  # Replace with your dataset path

data = load_data()

# Streamlit app
st.title("Sales Prediction App")

# Sidebar for user input
st.sidebar.header("User Input")
feature = st.sidebar.selectbox("Select the feature to train the model", ["TV", "Radio", "Newspaper"])

input_value = st.sidebar.number_input(f"Enter the value for {feature}")

if st.sidebar.button("Predict Sales"):
    # Select features and target
    X = data[[feature]]
    y = data["Sales"]
    
    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make prediction
    prediction = model.predict([[input_value]])
    
    st.write(f"Predicted Sales for {feature} = {input_value}  is : {prediction[0]:.2f}")
