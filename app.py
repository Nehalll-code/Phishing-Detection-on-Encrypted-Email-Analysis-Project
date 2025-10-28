from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import logging

app = Flask(__name__, template_folder="templates")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Define model paths
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
MODEL_PATH = "baseline_model.pkl"

# Check if model files exist
if not os.path.exists(VECTORIZER_PATH) or not os.path.exists(MODEL_PATH):
    error_msg = "Model files not found. Please run BaseLineModel.ipynb first."
    logger.error(error_msg)
    raise FileNotFoundError(error_msg)

# Load vectorizer & model
try:
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    logger.info("✅ Model and vectorizer loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading models: {str(e)}")
    raise

# 👇 Sample emails (some phishing, some safe)
sample_emails = [
    ("Your account has been suspended! Click here to verify.", "Phishing (Expected)"),
    ("Your Amazon order #12345 has been shipped. Thank you for shopping!", "Safe (Expected)"),
    ("Update your bank information immediately to avoid service disruption.", "Phishing (Expected)"),
    ("Meeting scheduled for tomorrow at 10 AM. Please confirm your attendance.", "Safe (Expected)"),
    ("Congratulations! You’ve won a $1000 gift card. Claim now!", "Phishing (Expected)")
]

@app.route("/", methods=["GET", "POST"])
def index():
    user_email = ""
    user_prediction = None
    user_confidence = None

    if request.method == "POST":
        try:
            # Get the email text from the form
            user_email = request.form["email_text"]
            
            # Convert input to TF-IDF features
            features = vectorizer.transform([user_email]).toarray()
            
            # Make prediction
            pred = model.predict(features)[0]
            prob = model.predict_proba(features)[0][pred]
            
            # Convert to boolean and percentage
            user_prediction = "Phishing" if pred == 1 else "Safe"
            user_confidence = round(prob * 100, 2)
            
            print(f"Prediction: {user_prediction}, Confidence: {user_confidence}%")
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return f"Error processing request: {str(e)}", 500

    return render_template(
        "index.html",
        user_email=user_email,
        user_prediction=user_prediction,
        user_confidence=user_confidence,
        sample_emails=sample_emails  # 👈 Pass sample emails to frontend
    )

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
