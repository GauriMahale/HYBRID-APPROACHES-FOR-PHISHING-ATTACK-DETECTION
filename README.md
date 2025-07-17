# 🛡️ HYBRID APPROACH FOR PHISHING ATTACK DETECTION

## 📌 Introduction

Phishing attacks pose a major threat to cybersecurity, yet detecting them is challenging due to their evolving tactics and high false positive rates.

To address these limitations, we propose a **hybrid phishing attack detection system** that combines the strengths of:

- ✅ Rule-based systems
- ✅ Machine Learning (ML)
- ✅ Deep Learning (DL)

Our system is evaluated on a real-world phishing URL dataset and has achieved **high accuracy** with **low false positives**. It is designed to detect even new and unknown phishing attempts **in real-time**, providing robust protection of users' personal and financial data.

![System Overview](https://github.com/GauriMahale/HYBRID-APPROACHES-FOR-PHISHING-ATTACK-DETECTION/assets/84667768/96e5e4e6-678c-4e3f-bc48-57116ecbfa98)

---

## 🎯 Objective

- Develop a **hybrid detection system** integrating rule-based, machine learning, and deep learning methods.
- Detect new and unknown phishing attacks **in real-time**.
- Reduce false positives while maintaining high detection accuracy.
- Protect users' **privacy, credentials, and financial data** from evolving phishing threats.

---

## 🔍 Methodology

The proposed system operates in three main phases:

1. **Data Preprocessing** – Load and clean phishing datasets using `pandas`.
2. **Feature Extraction** – Extract relevant features from URLs and HTML content.
3. **Classification** – Apply and compare multiple ML & DL models.

### Machine Learning Models Used:
- Gaussian Naive Bayes
- Support Vector Classifier (SVC)
- Extreme Learning Machine (ELM)
- Logistic Regression
- Random Forest
- **Ensemble Voting Classifier**

### Tech Stack:
- Python (pandas, sklearn, keras, Flask)
- Flask (for web deployment)
- Matplotlib / Seaborn (for metrics visualization)

---

### 🧠 System Workflow

![System Workflow](https://github.com/GauriMahale/HYBRID-APPROACHES-FOR-PHISHING-ATTACK-DETECTION/assets/84667768/d9dbfb00-9ca5-49eb-80cd-991dc3d25c7f)

**Fig 1.1 – System Workflow:**

- Dataset cleaning and preprocessing
- Feature selection from URLs and HTML
- Model training and testing
- Accuracy comparison
- Ensemble voting classifier for final prediction
- Flask-based web integration for user interaction

---

### 🧩 System Architecture

![Architecture](https://github.com/GauriMahale/HYBRID-APPROACHES-FOR-PHISHING-ATTACK-DETECTION/assets/84667768/a4b9a30e-4934-4b6a-83ed-1865e8bb6927)

**Fig 1.2 – System Architecture:**

- Input URLs go through feature extraction
- Features passed to multiple ML/DL models
- Prediction results combined using ensemble methods
- Results shown in web UI built with Flask

---

### 🔬 Proposed System

![Proposed System](https://github.com/GauriMahale/HYBRID-APPROACHES-FOR-PHISHING-ATTACK-DETECTION/assets/84667768/c97e4eaf-42c0-4551-9ce2-b56119f849d6)

**Fig 1.3 – Block Diagram of the Proposed System**

---

## 📊 Result

The hybrid approach yielded **high accuracy**, particularly when using the **Voting Classifier**, which aggregates predictions from individual models:

![Results](https://github.com/GauriMahale/HYBRID-APPROACHES-FOR-PHISHING-ATTACK-DETECTION/assets/84667768/2e415ac5-9484-4ac9-9ef9-277b65d4a662)

- **Accuracy:** 93%
- **Low False Positives**
- **Strong performance across unseen phishing data**

---

## ✅ Conclusion

The hybrid phishing detection system successfully integrates **deep learning** and **machine learning** models, offering robust performance against traditional and evolving phishing attacks.

- Achieved **93% accuracy** using a Voting Classifier.
- Ensures **real-time phishing detection**.
- Reduces manual intervention and false positives.
- Offers practical deployment via a Flask web interface.

---

## 👨‍💻 Author

**Gauri Mahale**  
Java Full-Stack Developer | AI/ML Enthusiast  

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 💬 Feedback / Contributions

Feel free to fork this repo, raise issues, or submit a pull request!

