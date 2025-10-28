# Phishing-Detection on Encrypted Email Analysis Project

## 📌 Project Overview  
This project implements a pipeline to detect phishing attempts in **encrypted emails** using machine-learning and homomorphic encryption techniques.  
It includes steps for data cleaning, feature extraction, model training, and a demo of encrypted email interception & classification.

## 🧠 Key Features  
- Data preprocessing: cleaning raw email data, extracting features from content and metadata.  
- Model training: baseline models (e.g., TF-IDF + classifier) to discriminate phishing vs legitimate emails.  
- Homomorphic encryption pipeline: demonstration of how email content can be encrypted and still used for classification without decryption.  
- Real-time demo: Server & client code to simulate encrypted email detection in action.  
- Full stack components: from data science notebooks to Python scripts & a live server/client setup.

## 📁 Repository Structure  
- `DataCleaning1.ipynb` — Notebook for email data cleaning and feature engineering.  
- `BaseLineModel.ipynb` — Notebook demonstrating baseline model implementation.  
- `homomorphicEncryptionPipeline.py` — Script implementing the encrypted-email model pipeline.  
- `homomorphicEncryptionDemo.py` — Demo script for encrypted-email inference.  
- `app.py` / `server.py` / `client.py` — Web/application interface for the system.  
- `requirements.txt` — Python dependencies.  
- `*.pkl`, `*.npy` files — Trained model artifacts & vectorizers.  
- `Cleaned_Phishing_Email.csv`, `cleaned_emails.csv` — Cleaned datasets.  
- `Phishing_Email.csv` — Original raw data.  

## 🚀 Getting Started  
### Prerequisites  
- Python 3.x  
- Install dependencies:  
  ```bash
  pip install -r requirements.txt
