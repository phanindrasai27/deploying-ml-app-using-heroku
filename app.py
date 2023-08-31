import streamlit as st
import pickle
import numpy as np

# Load the model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

# Set page title
st.set_page_config(page_title="ML API")

# Page header
st.title("Boston House Price Prediction")

# Input fields
CRIM = st.text_input("CRIM", value="0.0")
ZN = st.text_input("ZN", value="0.0")
INDUS = st.text_input("INDUS", value="0.0")
CHAS = st.text_input("CHAS", value="0")
NOX = st.text_input("NOX", value="0.0")
RM = st.text_input("RM", value="0.0")
Age = st.text_input("Age", value="0.0")
DIS = st.text_input("DIS", value="0.0")
RAD = st.text_input("RAD", value="0")
TAX = st.text_input("TAX", value="0.0")
PTRATIO = st.text_input("PTRATIO", value="0.0")
B = st.text_input("B", value="0.0")
LSTAT = st.text_input("LSTAT", value="0.0")

# Prediction button
if st.button("Predict"):
    # Prepare input for prediction
    input_data = np.array([
        float(CRIM), float(ZN), float(INDUS), int(CHAS), float(NOX),
        float(RM), float(Age), float(DIS), int(RAD), float(TAX),
        float(PTRATIO), float(B), float(LSTAT)
    ])

    # Scale input data
    scaled_input = scalar.transform(input_data.reshape(1, -1))

    # Perform prediction
    prediction = regmodel.predict(scaled_input)[0]

    # Display prediction result
    st.write(f"Predicted House Price: {prediction}")
