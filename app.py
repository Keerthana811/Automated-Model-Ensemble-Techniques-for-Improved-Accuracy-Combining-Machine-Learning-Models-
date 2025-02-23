import streamlit as st
import requests

BACKEND_URL = "http://localhost:5000"

st.title("Supermarket Sales Dashboard with Advanced Prediction")

# 🚀 Summary section
st.header("Summary Statistics")
sum_response = requests.get(f"{BACKEND_URL}/summary").json()
st.write(f"**Total Sales:** ${sum_response['total_sales']:.2f}")
st.write(f"**Average Rating:** {sum_response['avg_rating']:.2f}")
st.write(f"**Transactions:** {sum_response['num_transactions']}")

# 🔮 Prediction section
st.header("Predict Total Sales (Using XGBoost, Random Forest, and Stacking)")
quantity = st.number_input("Enter Quantity", min_value=1, value=1)
unit_price = st.number_input("Enter Unit Price", min_value=0.0, value=0.0, format="%.2f")

if st.button("Predict Total Sales"):
    payload = {"Quantity": quantity, "Unit price": unit_price}
    response = requests.post(f"{BACKEND_URL}/predict", json=payload).json()
    st.success(f"Predicted Total Sales: ${response['predicted_total_sales']:.2f}")
