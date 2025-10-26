# AI Agent Instructions for Phishing Detection with Privacy-Preserving ML

This codebase implements a privacy-preserving phishing email detection system using homomorphic encryption. Here's what you need to know to work effectively with this project.

## Architecture Overview

The system consists of three main components:

1. **Training Pipeline** (`BaseLineModel.ipynb`, `DataCleaning1.ipynb`)
   - Data preprocessing and TF-IDF feature extraction
   - Training logistic regression model for phishing detection
   - Model serialization via joblib

2. **Web Application** (`app.py`, `templates/index.html`)
   - Flask server for direct (non-encrypted) predictions
   - Simple web UI for email submission and results display

3. **Privacy-Preserving Inference** (`client.py`, `server.py`)
   - Client: Feature encryption using Paillier homomorphic encryption
   - Server: Secure inference on encrypted features without decryption
   - Communication via REST API endpoints

## Key Files

- `prepare_model.py`: Converts model weights to fixed-point integers for encryption
- `homomorphicEncryptionPipeline.py`: End-to-end example of encrypted inference
- `save_encrypted_emails.py`: Utility for creating encrypted email datasets
- `client.py`: Handles feature encryption and server communication
- `server.py`: Performs secure inference on encrypted data

## Important Patterns

1. **Fixed-Point Arithmetic**
   ```python
   SCALE = 10_000  # Must match between client/server
   ```
   All floating-point values are scaled to integers before encryption.

2. **Paillier Encryption Workflow**
   ```python
   # Client side
   public_key, private_key = paillier.generate_paillier_keypair()
   encrypted = public_key.encrypt(scaled_value)
   
   # Server side
   result = compute_on_encrypted(encrypted)  # No decryption needed
   ```

3. **Model Parameter Format**
   - Weights stored in `coef_scaled.npy`
   - Intercept stored in `intercept_scaled.npy`
   - Both use same scaling factor as features

## Development Setup

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Data flow sequence:
   - Run `DataCleaning1.ipynb` for data preprocessing
   - Run `BaseLineModel.ipynb` for model training
   - Run `prepare_model.py` to prepare for encryption
   - Start server with `python server.py`
   - Run client with `python client.py`

## Common Operations

- Feature extraction: Always use `tfidf_vectorizer.pkl` for consistency
- Web app testing: Run `python app.py` and visit http://127.0.0.1:5000
- Encrypted inference: Start server first, then run client script
- Dataset updates: Re-run notebooks in sequence, then `prepare_model.py`

## Tips

1. Check `scale` parameter matches between client and server
2. Use `homomorphicEncryptionDemo.py` for testing encryption workflow
3. Monitor memory usage with large feature vectors - encryption is resource-intensive
4. Web UI supports direct text input for quick testing