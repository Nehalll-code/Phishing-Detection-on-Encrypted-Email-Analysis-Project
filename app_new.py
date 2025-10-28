from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load vectorizer & model
try:
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("baseline_model.pkl")
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    raise

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
            user_prediction = bool(pred)
            user_confidence = round(prob * 100, 2)
            
            print(f"Prediction for email: {'Phishing' if user_prediction else 'Safe'}, Confidence: {user_confidence}%")
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return f"Error processing request: {str(e)}", 500

    return render_template(
        "index.html",
        user_email=user_email,
        user_prediction=user_prediction,
        user_confidence=user_confidence
    )

if __name__ == "__main__":
    app.run(debug=True)