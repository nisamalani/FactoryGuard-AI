from flask import Flask, request, jsonify
import joblib
import pandas as pd
import shap

app = Flask(__name__)

# Load model
model = joblib.load("models/model.pkl")

# SHAP explainer
explainer = shap.Explainer(model)

# Expected columns
EXPECTED_COLUMNS = [
    "vibration",
    "temperature",
    "pressure",
    "temp_roll_mean",
    "vibration_roll_mean",
    "temp_lag1",
    "vibration_lag1"
]


@app.route("/", methods=["GET"])
def home():
    return "FactoryGuard AI API is running 🚀"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # Ensure correct column order
        df = df[EXPECTED_COLUMNS]

        # Prediction
        pred = int(model.predict(df)[0])
        prob = float(model.predict_proba(df)[0][1])

        # SHAP explanation
        shap_values = explainer(df)

        values = shap_values.values
        if len(values.shape) == 3:
            values = values[0]

        feature_importance = dict(zip(df.columns, values[0]))

        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        top_features = [k for k, v in sorted_features[:2]]

        return jsonify({
            "failure_prediction": pred,
            "failure_probability": round(prob, 3),
            "top_factors": top_features
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)