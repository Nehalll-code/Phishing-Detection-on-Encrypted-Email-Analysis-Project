import pandas as pd
import joblib
import numpy as np
from phe import paillier
import csv

# === Load trained vectorizer & model ===
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("baseline_model.pkl")
print("✅ Vectorizer and model loaded.")

# === Load a subset of dataset (first 500 emails) ===
data = pd.read_csv("cleaned_emails.csv").head(20)   # 🔹 only first 500 emails
emails = data["Email Text"].tolist()
print(f"✅ Loaded {len(emails)} emails for demo.")

# === Convert → TF-IDF ===
X = vectorizer.transform(emails).toarray()
print("✅ TF-IDF features generated. Shape:", X.shape)

# === Limit number of features for encryption (for speed) ===
FEATURE_LIMIT = 20   # 🔹 use only first 20 TF-IDF features
X_small = X[:, :FEATURE_LIMIT]
print(f"✅ Using first {FEATURE_LIMIT} features for encryption.")

# === Generate Paillier keys ===
public_key, private_key = paillier.generate_paillier_keypair()
print("✅ Paillier keys generated.")

# === Encrypt subset ===
encrypted_rows = []
for idx, row in enumerate(X_small):
    enc_row = [public_key.encrypt(float(x)) for x in row]
    encrypted_rows.append(enc_row)
    if idx < 3:
        print(f"🔐 Sample encrypted row {idx+1}: {[str(e.ciphertext())[:25]+'...' for e in enc_row[:5]]}")

print("✅ Encryption complete for 500 emails.")

# === Save small preview for documentation ===
output_file = "encrypted_email_samples.csv"
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)  
    writer.writerow(["Email_Index", "Original_Email_Text", "Sample_Encrypted_Numbers"])
    for i, enc_row in enumerate(encrypted_rows[:5]):  # just 5 rows for preview
        sample_vals = [str(e.ciphertext()) for e in enc_row[:10]]
        writer.writerow([i+1, emails[i][:80], " | ".join(sample_vals)])

print(f"✅ Encrypted preview saved to {output_file}")
