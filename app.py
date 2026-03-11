import joblib
from flask import Flask, request, jsonify
import numpy as np
import shap
import time

app = Flask(__name__)
model = joblib.load("model.pkl")
explainer = shap.TreeExplainer(model)

@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    failure_prob = float(model.predict_proba(features)[0][1])
    prediction = int(failure_prob > 0.5)
    shap_vals = explainer.shap_values(features)[0]
    feature_names = data.get("feature_names", [f"feature_{i}" for i in range(len(shap_vals))])
    explanation = {name: round(float(val), 4) for name, val in zip(feature_names, shap_vals)}
    latency_ms = (time.time() - start) * 1000
    return jsonify({"failure_probability": round(failure_prob, 4), "prediction": "FAILURE_RISK" if prediction else "NORMAL", "shap_explanation": explanation, "latency_ms": round(latency_ms, 2)})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
