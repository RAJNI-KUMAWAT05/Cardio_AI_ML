from flask import Flask, request, jsonify, send_file
import joblib
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return send_file('index.html')

model = joblib.load("cardio_model.pkl")
model_columns = joblib.load("model_columns.pkl")

def get_alert(risk):
    if risk > 0.5:
        return "HIGH", "🚨 Immediate medical attention required"
    elif risk > 0.2:
        return "MEDIUM", "⚠️ Monitor regularly"
    else:
        return "LOW", "✅ Low risk"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    try:
        # 🔹 Convert input to DataFrame
        input_df = pd.DataFrame([data])

        # 🔹 Apply same encoding as training
        input_df = pd.get_dummies(input_df, drop_first=True)

        # 🔹 Align columns with model
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # 🔹 Convert boolean → int
        input_df = input_df.astype(int)

        print("FINAL INPUT TO MODEL:\n", input_df.to_dict())

        # 🔹 ML Prediction
        risk = model.predict_proba(input_df)[0][1]

        # 🔹 Base alert from ML
        if risk > 0.5:
            alert = "HIGH"
            message = "🚨 Immediate medical attention required"
            severity = "CRITICAL"
        elif risk > 0.2:
            alert = "MEDIUM"
            message = "⚠️ Monitor regularly"
            severity = "WARNING"
        else:
            alert = "LOW"
            message = "✅ Low risk"
            severity = "SAFE"

        # 🔥 Rule-based override (VERY IMPORTANT)
        if data.get('resting_blood_pressure', 0) > 180 and data.get('cholestoral', 0) > 300:
            alert = "HIGH"
            message = "🚨 Critical condition detected (High BP + Cholesterol)"
            severity = "CRITICAL"

        # 🔍 Reasons (Explainability)
        reasons = []

        if data.get('resting_blood_pressure', 0) > 140:
            reasons.append("High Blood Pressure")

        if data.get('cholestoral', 0) > 240:
            reasons.append("High Cholesterol")

        if data.get('Max_heart_rate', 0) < 120:
            reasons.append("Low Heart Rate")

        if data.get('exercise_induced_angina') == "Yes":
            reasons.append("Exercise Induced Angina")

        if data.get('oldpeak', 0) > 2:
            reasons.append("ST Depression (oldpeak)")

        print("Risk:", risk)

        # 🔹 Final response
        return jsonify({
            "risk": round(float(risk), 2),
            "alert": alert,
            "severity": severity,
            "message": message,
            "reasons": reasons
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
