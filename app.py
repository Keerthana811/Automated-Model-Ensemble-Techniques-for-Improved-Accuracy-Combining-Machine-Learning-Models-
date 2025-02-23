from flask import Flask, jsonify, request
from utils import load_data, get_summary, train_models, predict_total_sales
import joblib
import pandas as pd
from flask_cors import CORS
import os

app = Flask(__name__)

CORS(app)  # Enable CORS

data = load_data("../data/supermarket_sales.csv")
models = train_models(data)

MODEL_PATH = "../models/stacking_model.pkl"

# Load the trained model
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"🔥 Error loading the model: {e}")
        model = None
else:
    print("⚠️ Model file not found at:", MODEL_PATH)
    model = None


@app.route("/summary", methods=["GET"])
def summary():
    summary_data = get_summary(data)
    return jsonify(summary_data)

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.json
    prediction = predict_total_sales(models, input_data)
    return jsonify({"predicted_total_sales": prediction})

if __name__ == "__main__":
    app.run(debug=True)