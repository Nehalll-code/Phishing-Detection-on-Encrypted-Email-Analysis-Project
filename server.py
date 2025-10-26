# server.py
from flask import Flask, request, jsonify
import numpy as np
#import paillier as _dummy  # placeholder for typing; we'll import from phe
from phe import paillier
import json

app = Flask(__name__)

# Load scaled model parameters
coef = np.load("coef_scaled.npy")            # int array
intercept = int(np.load("intercept_scaled.npy")[0])  # int

# NOTE: coef length should match TF-IDF vector length used by client
print("Server: loaded coef length:", len(coef))

def deserialize_encrypted(public_key, enc_obj):
    # enc_obj: dict { "c": "<ciphertext as decimal string>", "e": exponent (int) }
    c = int(enc_obj["c"])
    e = int(enc_obj["e"])
    return paillier.EncryptedNumber(public_key, c, e)

@app.route("/predict_enc", methods=["POST"])
def predict_enc():
    payload = request.get_json()
    # must contain pub_n (string), enc_vec (list), scale (int)
    pub_n = int(payload["pub_n"])
    enc_vec_serialized = payload["enc_vec"]
    scale = int(payload["scale"])

    public_key = paillier.PaillierPublicKey(pub_n)

    # Reconstruct the encrypted feature vector
    enc_vec = [deserialize_encrypted(public_key, enc_obj) for enc_obj in enc_vec_serialized]

    if len(enc_vec) != len(coef):
        return jsonify({"error": f"vector length mismatch: got {len(enc_vec)}, expected {len(coef)}"}), 400

    # Compute encrypted linear score: Enc( sum_i coef_i * x_i + intercept )
    # Strategy: for each feature, compute enc_vec[i] ** coef_i (scalar multiply)
    # then multiply all those EncryptedNumbers together (which corresponds to sum of terms)
    # finally multiply by public_key.encrypt(intercept)
    enc_sum = public_key.encrypt(0)  # encryption of zero as starting value

    # Multiply-accumulate using homomorphic properties
    # EncryptedNumber.__pow__ does scalar multiplication (EncryptedNumber ** k).
    # EncryptedNumber __mul__ adds plaintexts (because ciphertexts multiply).
    for i, w in enumerate(coef):
        if w == 0:
            continue
        # w is integer (can be negative). EncryptedNumber.__pow__ supports negative ints as multiplication by -1^?
        enc_term = enc_vec[i].__pow__(int(w))  # scalar multiply
        enc_sum = enc_sum * enc_term

    # add intercept by multiplying with encryption of intercept
    enc_intercept = public_key.encrypt(int(intercept))
    enc_sum = enc_sum * enc_intercept

    # Serialize return: ciphertext (decimal string) and exponent
    ser = {"c": str(enc_sum.ciphertext()), "e": enc_sum.exponent}
    return jsonify({"enc_score": ser})

@app.route("/", methods=["GET"])
def home():
    return "Flask server is running!"


if __name__ == "__main__":
    # run on localhost:5000
    app.run(host="127.0.0.1", port=5000, debug=True)
