from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_from_directory
from sklearn.exceptions import InconsistentVersionWarning

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
DATA_PATH = BASE_DIR / "Mall_Customers.csv"


def load_model():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        with MODEL_PATH.open("rb") as f:
            return pickle.load(f)


model_bundle = load_model()


def load_customer_stats():
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception:
        return {
            "customers": 0,
            "avg_age": 0,
            "avg_income": 0,
            "avg_score": 0,
        }

    return {
        "customers": len(df),
        "avg_age": round(df["Age"].mean(), 1),
        "avg_income": round(df["Annual Income (k$)"].mean(), 1),
        "avg_score": round(df["Spending Score (1-100)"].mean(), 1),
    }


CUSTOMER_STATS = load_customer_stats()


def build_features(age, income, gender):
    if isinstance(model_bundle, dict):
        scaler = model_bundle.get("scaler")
        label_encoder = model_bundle.get("label_encoder")

        raw_values = pd.DataFrame(
            [[age, income]],
            columns=["Age", "Annual Income (k$)"],
        )
        scaled_values = scaler.transform(raw_values)[0] if scaler else [age, income]
        encoded_gender = (
            label_encoder.transform([gender])[0]
            if label_encoder is not None
            else 1 if gender == "Male" else 0
        )
        return pd.DataFrame(
            [[scaled_values[0], scaled_values[1], encoded_gender]],
            columns=["Age_Scaled", "Income_Scaled", "Genre_Encoded"],
        )

    return np.array([[age, income, 1 if gender == "Male" else 0]])


def predict_spending_score(age, income, gender):
    features = build_features(age, income, gender)
    estimator = model_bundle.get("model") if isinstance(model_bundle, dict) else model_bundle
    score = float(estimator.predict(features)[0])
    return round(max(1, min(100, score)), 1)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    form_values = {
        "age": request.form.get("age", "32"),
        "income": request.form.get("income", "65"),
        "gender": request.form.get("gender", "Female"),
    }

    if request.method == 'POST':
        try:
            age = float(form_values["age"])
            income = float(form_values["income"])
            gender = form_values["gender"]

            if not 1 <= age <= 120:
                raise ValueError("Age must be between 1 and 120.")
            if not 0 <= income <= 300:
                raise ValueError("Annual income must be between 0 and 300 k$.")
            if gender not in {"Female", "Male"}:
                raise ValueError("Choose a valid gender value.")

            prediction = predict_spending_score(age, income, gender)

        except Exception as e:
            error = str(e)

    return render_template(
        'index.html',
        prediction=prediction,
        error=error,
        form_values=form_values,
        stats=CUSTOMER_STATS,
    )


@app.route('/assets/<path:filename>')
def assets(filename):
    return send_from_directory(BASE_DIR, filename)


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5001, debug=False)
