# 💳 Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Optimization-green)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📌 Project Overview

This project develops an end-to-end machine learning pipeline for detecting fraudulent credit card transactions using the Credit Card Fraud Detection dataset.

The dataset is highly imbalanced, making fraud detection a challenging classification problem. The project focuses on selecting appropriate evaluation metrics, handling class imbalance, optimizing model performance, and improving decision-making through threshold optimization.

---

## 🎯 Key Takeaways

- Detect fraudulent credit card transactions
- Compare multiple machine learning algorithms
- Handle severe class imbalance
- Optimize model hyperparameters using Optuna
- Improve classification performance through threshold tuning
- Track experiments using MLflow

---

## 🎯 Problem Statement

This project aims to:
- Build a robust classification model
- Compare multiple ML algorithms
- Optimize the best-performing model
- Provide model interpretability using SHAP

---
## 🔄 Workflow

```text
EDA
   ↓
Feature Selection
   ↓
Train / Test Split
   ↓
Model Benchmarking
(Logistic, RF, XGBoost, LightGBM)
   ↓
Best Model → Random Forest
   ↓
Optuna Hyperparameter Optimization
   ↓
Threshold Optimization
   ↓
Final Evaluation
   ↓
Feature Importance / SHAP
   ↓
MLflow Tracking
   ↓
Save Model
```
---

## 📂 Project Structure

```text
Credit_Card_Fraud_Detection/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_Modeling.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── tune.py
│   ├── evaluate.py
│   └── save_model.py
│
├── scripts/
│   └── run_training.py
│
├── models/
│
├── requirements.txt
└── README.md
```

---

## 📊 Exploratory Data Analysis

The EDA focused on:

- Class imbalance analysis
- Distribution of numerical features
- Correlation with the target variable
- Statistical analysis of PCA-transformed features
- Feature selection using baseline experiments

Since most predictors are PCA-transformed components (`V1–V28`), the analysis emphasizes statistical behavior rather than semantic interpretation.

---

## 🤖 Models Benchmarking

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

Model comparison was performed using Stratified Cross Validation.

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score

✔ **Best Model: RandomForest**

**why Random Forest?**
Random Forest was selected as the final model because it consistently outperformed the other algorithms during benchmarking. Its performance was further improved through Optuna-based hyperparameter optimization and threshold tuning.

---

## ⚖️ Handling Class Imbalance

The dataset contains a very small percentage of fraudulent transactions.

To address this issue:

- Stratified Train/Test Split
- SMOTE applied only on the training set
- F1-score used as the primary optimization metric
- Threshold optimization instead of relying on the default probability threshold (0.5)
Although SMOTE is a widely used technique for handling class imbalance, it did not improve performance on this dataset. Due to the severe performance degradation observed after applying SMOTE, the final model was trained on the original imbalanced dataset. Threshold optimization was used instead to improve minority class detection.
---

## ⚙️ Hyperparameter Optimization

Hyperparameters were optimized using **Optuna** with cross-validation.

The best configuration was then retrained on the full training dataset.

### Tuned parameters:
- max_depth
- n_estimators

  ✔ Optimization improved model stability and generalization.
---

## 📈 Threshold Optimization

Instead of using the default decision threshold of **0.5**, the optimal threshold was selected by maximizing the F1-score based on predicted probabilities.

This improved fraud detection performance while maintaining a balance between precision and recall.

---

## 📊 Model Performance

| Metric | Score |
|---------|------:|
| Accuracy | 0.999684 |
| Precision | 0.954545 |
| Recall | 0.857143 |
| F1-score | 0.903226 |

### Confusion Matrix
<img width="524" height="432" alt="confus" src="https://github.com/user-attachments/assets/110065d7-34ac-46f6-bf1c-5d7725137927" />

### ROC Curve
<img width="790" height="590" alt="ROC" src="https://github.com/user-attachments/assets/b145456c-f813-4fb5-b4d4-38cd09eabb28" />

### Precision-Recall Curve
<img width="789" height="590" alt="pres recall" src="https://github.com/user-attachments/assets/932716bd-c439-4ed8-92e3-b12bd5e8a656" />

---

## 🧠 Model Explainability

Feature importance / SHAP analysis was used to understand the contribution of individual features to fraud prediction.

### SHAP Summary Plot
<img width="748" height="540" alt="SHAP" src="https://github.com/user-attachments/assets/daa1c803-b284-4c1e-ba02-c05924e43fc3" />

---

## 📋 Experiment Tracking

Experiments were tracked using MLflow, including:

- Hyperparameters
- Cross-validation scores
- Evaluation metrics
- Trained models

To launch UI:

```bash
mlflow ui
```

---

## 🛠 Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Optuna
- MLflow
- SHAP
- Matplotlib
- Seaborn

---

## 🚀 How to Run

```bash
pip install -r requirements.txt

python scripts/run_training.py
```

---

## 📚 Lessons Learned

- Why accuracy is misleading for imbalanced datasets.
- Importance of Precision-Recall metrics in fraud detection.
- Effect of threshold optimization on minority class detection.
- Although SMOTE is a widely used technique for handling class imbalance, it did not improve performance on every dataset.

## 🚀 Future Work

Possible directions for extending this project include:

- Evaluate additional imbalance-aware learning techniques such as Balanced Random Forest, EasyEnsemble, or cost-sensitive learning.
- Investigate anomaly detection methods (e.g., Isolation Forest or One-Class SVM) for identifying previously unseen fraud patterns.
- Optimize the decision threshold using business-driven cost functions instead of maximizing only the F1-score.
- Deploy the trained model as a REST API using FastAPI for real-time fraud detection.
- Build an interactive dashboard to monitor prediction performance and model drift in production.
- Explore automated retraining and continuous experiment tracking using MLOps tools.
