# 💖 Heart Analyzer
Heart Analyzer is a modern, **Streamlit-based AI web app** that predicts the risk of heart disease using medical parameters.  
It uses a **Support Vector Machine (SVM)** model trained on clinical data and provides clear, user-friendly health insights.  

---

## 💼 Features
-  **Health Parameter Input** – Enter age, blood pressure, cholesterol, and other metrics.
-  **AI-Powered Prediction** – Uses an SVM classifier to detect heart disease risk.
-  **Health Score**🧬 – Displays a percentage-based score for overall heart health.
-  **Key Risk Factors** 🔎– Identifies likely contributing factors (cholesterol, chest pain type, etc.).
- **Medical Guidance Sidebar** – Explains each parameter in simple terms.
- **Instant Feedback** – Shows whether the user is at risk and suggests next steps.

---

## 🌐 Use Cases
- **Healthcare Professionals** – Quick screening support tool.  
- **Students & Researchers** – Educational project for ML in healthcare.  
- **General Users** – Personal health awareness and risk assessment.  

---

## 🔒 Privacy & Disclaimer
- All predictions are computed locally on your system.  
- Your data is **not stored** or shared.  
- ⚠️ **Disclaimer**: This app is for educational & awareness purposes only. It is **not a substitute for professional medical advice**.  

---

## 📦 Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/Dharshiniramdj/heart-analyzer.git
cd heart-analyzer
pip install -r requirements.txt
```

---

## ▶️ Run the App
```bash
streamlit run heart_analyzer.py
```

---

## 📂 Project Structure
```
heart-analyzer/
├── heart_analyzer.py      # Main Streamlit app
├── heart.csv              # Dataset used for training & testing
├── DATASET_FEATURES.md    # Explanation of dataset columns
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
└── LICENSE                # License file
```

---

## 📚 Tech Stack
- **Framework:** Streamlit  
- **Machine Learning:** scikit-learn (SVM)  
- **Data Handling:** Pandas  
- **UI Styling:** Streamlit components  

---

## 📄 License
Licensed under the MIT License.

---

👩‍💻 **Author** 

Dharshini Ram
Capstone Project – *Artificial intelligence (CSA1706)*  
AI & Data Science enthusiast passionate about building health-focused, accessible tools.
