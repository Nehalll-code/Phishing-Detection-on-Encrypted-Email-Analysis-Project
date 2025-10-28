# 🛡️ Phishing Email Detection

[![Python Version](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![GitHub Issues](https://img.shields.io/github/issues/Nehalll-code/Phishing-Detection-on-Encrypted-Email-Analysis-Project)](https://github.com/Nehalll-code/Phishing-Detection-on-Encrypted-Email-Analysis-Project/issues)
[![Status](https://img.shields.io/badge/Status-Experimental-orange)](#)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen)](#)

---

## 🔍 Project Overview
This project is a **machine learning-based phishing email detection system**. It can classify emails as **phishing or legitimate** using text-based features such as TF-IDF vectorization.  

**Key Goals:**
- Detect phishing attempts in email content  
- Serve as a research & educational tool for email security  
- Provide baseline ML implementation for phishing detection projects  

---

## ⚙️ Tech Stack
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)  
[![NumPy](https://img.shields.io/badge/NumPy-1.27-orange)](https://numpy.org/)  
[![Pandas](https://img.shields.io/badge/Pandas-2.1-blueviolet)](https://pandas.pydata.org/)  
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-2.2-green)](https://scikit-learn.org/)  
[![Joblib](https://img.shields.io/badge/Joblib-1.3-lightblue)](https://joblib.readthedocs.io/)

---

## 🚀 Installation & Setup
```bash
# Clone the repository
git clone https://github.com/Nehalll-code/Phishing-Detection-on-Encrypted-Email-Analysis-Project.git
cd Phishing-Detection-on-Encrypted-Email-Analysis-Project

# Install dependencies
pip install -r requirements.txt

🏃‍♂️ Usage
1️⃣ Data Cleaning & Preprocessing
jupyter notebook DataCleaning1.ipynb
2️⃣ Train or Load Model
jupyter notebook BaseLineModel.ipynb


Or load pre-trained model:

import joblib
model = joblib.load("baseline_model.pkl")

3️⃣ Predict Emails
python predict_emails.py

📦 Dataset

Source: Provided CSV files (Phishing_Email.csv)

Content: Raw email texts labeled as phishing or legitimate

Usage: Preprocess using DataCleaning1.ipynb before training/testing
