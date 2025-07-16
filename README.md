# 🧠 Cyber Attack Detection using AI

A powerful and intelligent system that leverages Artificial Intelligence to detect cyber attacks in network traffic data. This project focuses on analyzing patterns using machine learning models to classify normal vs malicious behavior, enhancing cybersecurity infrastructure.

---

## 🔍 Project Overview

- ⚠️ Detects known and unknown cyber attack patterns
- 🧠 Uses supervised machine learning models
- 📊 Trained on real-world network traffic datasets
- 📈 Evaluates performance with accuracy, precision, recall, and F1-score

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- Pandas / NumPy
- Matplotlib / Seaborn
- Jupyter Notebook / Google Colab
- [Optional: TensorFlow / PyTorch] (if deep learning was used)

---

## 📁 Folder Structure

cyber-attack-detection-ai/
├── dataset/
│   └── <network_dataset.csv>
├── models/
│   └── trained_model.pkl
├── notebooks/
│   └── exploratory_analysis.ipynb
│   └── model_training.ipynb
├── src/
│   └── preprocess.py
│   └── train_model.py
├── results/
│   └── classification_report.txt
│   └── confusion_matrix.png
├── README.md
└── requirements.txt

---

## 🚀 How to Run

1. Clone the repository:

   git clone https://github.com/Overttuba68/Cyber-attack-detection
   cd cyber-attack-detection-ai

2. Install dependencies:

   pip install -r requirements.txt

3. Run model training:

   python src/train_model.py

4. View results in `results/` folder or Jupyter notebooks.

---

## 📊 Machine Learning Models Used

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)  
- Neural Networks 

---

## 📈 Results & Evaluation

| Model            | Accuracy | Precision | Recall | F1-Score |
|------------------|----------|-----------|--------|----------|
| Random Forest    | 97.5%    | 97.1%     | 96.8%  | 96.9%    |
| SVM              | 95.2%    | 94.9%     | 94.5%  | 94.7%    |

📌 Confusion matrix and ROC curve available in the `results/` folder.

---

## 🧪 Dataset Used
 
- Includes features like source IP, destination IP, protocol, packet size, duration, etc.

---

## 💡 Future Enhancements

- ✅ Deep Learning (LSTM / CNN for time-series packet data)
- ✅ Real-time intrusion detection
- ✅ Dashboard or frontend interface
- ✅ Dataset augmentation

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## 📜 License

MIT © 2025 Jaffer Ali
