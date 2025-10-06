from flask import Flask, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load vectorizer and model (only once)
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("baseline_model.pkl")

# Sample emails
emails = [
    "Congratulations! You have won a $1000 gift card.",
    "Please find attached the meeting agenda for tomorrow.",
    "Urgent: Your account has been compromised. Reset your password now.",
    "Hey, just wanted to check if you're free for lunch.",
    "Limited time offer! Get 70% off on all products today."
]

# Transform and predict (no encryption for demo)
X = vectorizer.transform(emails).toarray()
preds = model.predict(X)

@app.route('/')
def home():
    results = list(zip(emails, preds))
    return render_template('index.html', results=results)

if __name__ == "__main__":
    app.run(debug=True)
