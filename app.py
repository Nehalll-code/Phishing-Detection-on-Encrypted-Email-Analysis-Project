from flask import Flask, render_template, request
import joblib
import numpy as np
import sklearn  # Add this import to handle the warning

app = Flask(__name__)

# Load vectorizer & model with explicit return_type
try:
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("baseline_model.pkl")
    print("✅ Model and vectorizer loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {str(e)}")
    raise

@app.route("/", methods=["GET", "POST"])
def index():
    user_email = ""
    user_prediction = None
    user_confidence = None

    if request.method == "POST":
        user_email = request.form["email_text"]

        # Convert input to TF-IDF features
        features = vectorizer.transform([user_email]).toarray()

        # Predict phishing (1) or not (0)
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][pred]

        user_prediction = bool(pred)
        user_confidence = round(prob * 100, 2)

    return render_template(
        "index.html",
        user_email=user_email,
        user_prediction=user_prediction,
        user_confidence=user_confidence
    )

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
