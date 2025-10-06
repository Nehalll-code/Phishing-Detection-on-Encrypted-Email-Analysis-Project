import numpy as np
from phe import paillier
import joblib  # use joblib instead of pickle

# === Load vectorizer and model ===
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("baseline_model.pkl")
print("✅ Vectorizer and model loaded.")

# Sample emails
emails = [
    "Congratulations! You have won a $1000 gift card. Click here to claim now.",
    "Please find attached the meeting agenda for tomorrow's team discussion.",
    "Urgent: Your account has been compromised. Reset your password immediately.",
    "Hey, just wanted to check if you're free for lunch tomorrow.",
    "Limited time offer! Get 70% off on all products today only."
]

# Generate TF-IDF features
X = vectorizer.transform(emails).toarray()  # shape: (n_emails, 5000)
print(f"✅ TF-IDF features generated. Shape: {X.shape}")

# Generate Paillier keys
public_key, private_key = paillier.generate_paillier_keypair()
print("✅ Paillier keys generated.")

# Encrypt top N features
N = 50
encrypted_data = []
for row in X:
    encrypted_row = [public_key.encrypt(float(x)) for x in row[:N]] + list(row[N:])
    encrypted_data.append(encrypted_row)
print(f"✅ Encryption done — first {N} features encrypted.")

# Decrypt before prediction
decrypted_data = []
for row in encrypted_data:
    decrypted_row = [private_key.decrypt(x) if i < N else x for i, x in enumerate(row)]
    decrypted_data.append(decrypted_row)

decrypted_data = np.array(decrypted_data)
print("✅ Decryption done — ready for prediction.")

# Make predictions
preds = model.predict(decrypted_data)
for email, pred in zip(emails, preds):
    print("Email:", email[:50], "...")
    print("Prediction:", "Phishing" if pred else "Not Phishing", "\n")
