# client.py
import joblib
import numpy as np
from phe import paillier
import requests
import json

VECT_PATH = "tfidf_vectorizer.pkl"
SCALE = 10_000  # must match prepare_model.py scale
SERVER_URL = "http://127.0.0.1:5000/predict_enc"

# Load vectorizer
vectorizer = joblib.load(VECT_PATH)

# Example emails to test
emails = [
    "Congratulations! You have won a $1000 gift card. Click here to claim now.",
    "Please find attached the meeting agenda for tomorrow's team discussion.",
    "Urgent: Your account has been compromised. Reset your password immediately.",
]

# Create TF-IDF feature vectors (dense)
X = vectorizer.transform(emails).toarray()  # shape: (n_emails, n_features)

# Generate Paillier keypair (client keeps private_key)
public_key, private_key = paillier.generate_paillier_keypair(n_length=2048)
print("Client: generated Paillier keypair.")

def encrypt_feature_vector(public_key, float_vec, scale):
    # float_vec: 1D numpy array of floats
    # scale -> convert to integers for fixed point
    scaled = np.round(float_vec * scale).astype("int64")
    enc_vec = []
    for val in scaled:
        # encrypt integer; using public_key.encrypt produces an EncryptedNumber
        enc = public_key.encrypt(int(val))
        enc_vec.append({"c": str(enc.ciphertext()), "e": enc.exponent})
    return enc_vec

def sigmoid(x):
    import math
    return 1 / (1 + math.exp(-x))

# For each email: encrypt vector, POST to server, decrypt response
for i, row in enumerate(X):
    enc_vec_serialized = encrypt_feature_vector(public_key, row, SCALE)
    payload = {
        "pub_n": str(public_key.n),
        "scale": SCALE,
        "enc_vec": enc_vec_serialized
    }
    r = requests.post(SERVER_URL, json=payload, timeout=30)
    if r.status_code != 200:
        print("Server error:", r.text)
        continue
    res = r.json()
    enc_score = res["enc_score"]
    # reconstruct encrypted number on client side to decrypt:
    c = int(enc_score["c"])
    e = int(enc_score["e"])
    from phe import paillier as _p
    enc_number = paillier.EncryptedNumber(public_key, c, e)
    # decrypt
    score_scaled = private_key.decrypt(enc_number)  # integer representing SCALE * (w·x + b)
    score = score_scaled / SCALE
    prob = sigmoid(score)
    label = "Phishing" if prob >= 0.5 else "Not Phishing"
    print(f"Email {i}: score={score:.4f}, prob={prob:.4f} -> {label}")
