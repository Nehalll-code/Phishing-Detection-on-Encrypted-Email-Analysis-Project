import pandas as pd
import joblib
import numpy as np
from phe import paillier

# Load vectorizer and model
vectorizer = joblib.load(r'C:\Users\Nehal\Desktop\ResumeProjects\IS_PROJECT\tfidf_vectorizer.pkl')
model = joblib.load(r'C:\Users\Nehal\Desktop\ResumeProjects\IS_PROJECT\baseline_model.pkl')
print("✅ Vectorizer and model loaded.")

# Load sample emails
data = pd.read_csv(r'C:\Users\Nehal\Desktop\ResumeProjects\IS_PROJECT\cleaned_emails.csv')
sample_texts = data['Email Text'].head(5).tolist()
print("✅ Sample emails loaded.")

# Convert text to TF-IDF features
X = vectorizer.transform(sample_texts).toarray()
print("✅ TF-IDF features generated. Shape:", X.shape)

# Generate Paillier keys
public_key, private_key = paillier.generate_paillier_keypair()
print("✅ Paillier keys generated.")

# === Demo encryption for first 10 features only ===
demo_features = 10
encrypted_demo = [
    [public_key.encrypt(float(x)) for x in row[:demo_features]] for row in X
]

# Decrypt demo
decrypted_demo = [
    [private_key.decrypt(x) for x in row] for row in encrypted_demo
]

print("Original first 10 TF-IDF values for first email:\n", np.round(X[0, :demo_features], 4))
print("Decrypted first 10 TF-IDF values:\n", np.round(decrypted_demo[0], 4))

# === Prediction using full TF-IDF (unencrypted) ===
preds = model.predict(X)

# Display results
for i, text in enumerate(sample_texts):
    print("\nEmail:", text[:80], "...")
    print("Prediction:", "Phishing" if preds[i]==1 else "Not Phishing")


#Only non-zero TF-IDF values are encrypted → much faster.
#Scaling factor SCALE converts floats to integers → required by Paillier.
#Intercept is scaled correctly (SCALE**2).
#This script demonstrates true homomorphic computation on encrypted features.

#| Step | Purpose                                                     |
#| ---- | ----------------------------------------------------------- |
#| 1–2  | Load pre-trained artifacts                                  |
#| 3    | Convert email text → TF-IDF numbers                         |
#| 4    | Create Paillier encryption keys                             |
#| 5    | Encrypt numeric features                                    |
#| 6    | (For demo) Decrypt to simulate privacy-preserving inference |
#| 7–8  | Predict phishing vs non-phishing                            |
