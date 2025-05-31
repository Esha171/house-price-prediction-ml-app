import streamlit as st
import requests

st.title("🏠 House Price Prediction App")

st.write("Enter the house features:")

area = st.number_input("Area", value=5000)
bedrooms = st.number_input("Bedrooms", value=3)
bathrooms = st.number_input("Bathrooms", value=2)
stories = st.number_input("Stories", value=1)
mainroad = st.selectbox("Main Road?", ["No", "Yes"])
guestroom = st.selectbox("Guest Room?", ["No", "Yes"])
basement = st.selectbox("Basement?", ["No", "Yes"])
hotwaterheating = st.selectbox("Hot Water Heating?", ["No", "Yes"])
airconditioning = st.selectbox("Air Conditioning?", ["No", "Yes"])
parking = st.number_input("Parking Spots", value=1)
prefarea = st.selectbox("Preferred Area?", ["No", "Yes"])
furnishingstatus = st.selectbox("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Furnished"])

if st.button("Predict Price"):
    data = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": 1 if mainroad == "Yes" else 0,
        "guestroom": 1 if guestroom == "Yes" else 0,
        "basement": 1 if basement == "Yes" else 0,
        "hotwaterheating": 1 if hotwaterheating == "Yes" else 0,
        "airconditioning": 1 if airconditioning == "Yes" else 0,
        "parking": parking,
        "prefarea": 1 if prefarea == "Yes" else 0,
        "furnishingstatus": 0 if furnishingstatus == "Unfurnished" else 1 if furnishingstatus == "Semi-Furnished" else 2
    }
    res = requests.post("http://127.0.0.1:8000/predict", json=data)
    st.success(f"Predicted Price: ₹{int(res.json()['predicted_price'])}")