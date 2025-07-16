# 🔒 Cyber Attack Detection

A machine-learning system that detects cyber attacks in network traffic.  
Classifies normal and malicious traffic into categories (DoS, Probe, R2L, U2R).

---

## 🚀 Features
✅ Preprocesses network traffic data  
✅ Trains ML models for classification  
✅ Evaluates models with accuracy, confusion matrix, ROC curve  
✅ Saves trained model for future use  

---

## 🔗 Dataset
Uses [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) / [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) dataset.  
Include a small sample in `dataset/sample_data.csv`.

---

## ⚙️ Installation
```bash
git clone https://github.com/YOUR_USERNAME/cyber-attack-detection.git
cd cyber-attack-detection
pip install -r requirements.txt
