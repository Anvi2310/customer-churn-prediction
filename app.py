from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and feature names
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
feature_names = model_data["features_names"]

# Load encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get all form inputs
    user_input = {}
    for feature in feature_names:
        user_input[feature] = request.form.get(feature)

    input_df = pd.DataFrame([user_input])

    try:
        # Encode categorical features
        for col in input_df.columns:
            if col in encoders:
                input_df[col] = encoders[col].transform(input_df[col])

        # Convert to float
        input_df = input_df.astype(float)
    except (ValueError, TypeError):
        return render_template(
            "index.html",
            prediction_text="Please fill in every field with a valid value.",
        )

    # Predict
    prediction = model.predict(input_df)[0]
    result = "Yes (Customer is likely to churn)" if prediction == 1 else "No (Customer is likely to stay)"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)