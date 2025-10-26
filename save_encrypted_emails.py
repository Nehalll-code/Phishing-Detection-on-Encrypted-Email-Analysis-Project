import pandas as pd
import joblib
import numpy as np
from phe import paillier
import csv

# === Load trained vectorizer & model ===
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("baseline_model.pkl")
print("✅ Vectorizer and model loaded.")

# === Load dataset ===
data = pd.read_csv("cleaned_emails.csv")
emails = data["Email Text"].tolist()
print(f"✅ Loaded {len(emails)} emails.")

# === Convert all emails → TF-IDF ===
X = vectorizer.transform(emails).toarray()
print("✅ TF-IDF features generated. Shape:", X.shape)

# === Generate Paillier keys ===
public_key, private_key = paillier.generate_paillier_keypair()
print("✅ Paillier keys generated.")

# === Encrypt all features (or first 100 for speed) ===
N = 100  # change to 5000 if you want full encryption
encrypted_rows = []
for idx, row in enumerate(X):
    enc_row = [public_key.encrypt(float(x)) for x in row[:N]]
    encrypted_rows.append(enc_row)
    if idx < 3:
        print(f"🔐 Sample encrypted row {idx+1}: {[str(e.ciphertext())[:25]+'...' for e in enc_row[:5]]}")

# === Save a “visualization” CSV for your professor ===
output_file = "encrypted_email_samples.csv"
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Email_Index", "Original_Email_Text", "Sample_Encrypted_Numbers"])
    for i, enc_row in enumerate(encrypted_rows[:5]):  # just show 5 emails for readability
        sample_vals = [str(e.ciphertext()) for e in enc_row[:10]]
        writer.writerow([i+1, emails[i][:80], " | ".join(sample_vals)])

print(f"✅ Encrypted preview saved to {output_file}")
