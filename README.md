# 🚗 Insurance Claim Risk Prediction using Machine Learning

## 📌 Project Overview

Insurance companies face major financial risks when high-risk customers frequently make insurance claims.  
Identifying such risky policyholders early helps companies make better underwriting and risk assessment decisions.

This project uses Machine Learning classification techniques to predict whether a customer is likely to make an insurance claim based on customer, vehicle, and policy-related information.

The complete pipeline includes:

- Data preprocessing
- Feature engineering
- Imbalance handling
- Model training
- Cross validation
- Hyperparameter tuning
- Threshold optimization
- Final deployment using Streamlit & Hugging Face

The final model was built using **XGBoost with Class Weights**, which provided the best business-oriented performance for identifying risky customers.

---

# 🎯 Business Problem

Insurance claim datasets are highly imbalanced because only a small percentage of customers actually make claims.

Traditional rule-based systems often fail to identify risky policyholders accurately, which can lead to:

- Increased financial losses
- Poor underwriting decisions
- Incorrect insurance approvals
- Weak risk management strategies

The objective of this project is to build a Machine Learning system capable of identifying potentially risky customers before insurance approval.

---

# 🧠 Machine Learning Objective

The model predicts:

- **0 → No Claim**
- **1 → Claim**

Instead of focusing only on accuracy, the project emphasizes:

- Recall
- F1-Score
- ROC-AUC
- Business-oriented risk detection

because detecting risky customers is more important than simply achieving high accuracy on majority-class data.

---

# 📂 Dataset Information

- Total Records: **58,592**
- Features: **41**
- Problem Type: **Binary Classification**
- Highly Imbalanced Dataset:
  - No Claim → ~93.4%
  - Claim → ~6.6%

---

# ⚙️ Technologies Used

## Programming & Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn

## Deployment

- Streamlit
- Hugging Face Spaces

---

# 🛠️ Project Workflow

## 1️⃣ Data Preprocessing

- Removed unnecessary columns
- Extracted numerical values from text-based vehicle features
- Handled categorical variables
- Created engineered features
- Performed stratified train-test split
- Built preprocessing pipelines using:
  - StandardScaler
  - OneHotEncoder
  - OrdinalEncoder
  - SimpleImputer

---

# 🎥 Project Demo

https://github.com/user-attachments/assets/cc7f1f42-5599-4ca2-9bed-6abf8ec3a2cd

---


## 2️⃣ Imbalance Handling

The dataset imbalance was handled using:

- Baseline Training
- Class Weight Balancing
- SMOTE Oversampling

---

## 3️⃣ Model Training

The following classification models were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- Gaussian Naive Bayes
- Gradient Boosting
- XGBoost

---

## 4️⃣ Cross Validation

Stratified Cross Validation was used to:

- maintain imbalance distribution across folds
- evaluate model stability
- detect overfitting
- improve generalization performance

---

## 5️⃣ Hyperparameter Tuning

RandomizedSearchCV was used to optimize:

- max_depth
- learning_rate
- n_estimators
- gamma
- subsample
- regularization parameters

This significantly reduced overfitting and improved model stability.

---

## 6️⃣ Threshold Optimization

Threshold tuning was performed to find the most business-effective balance between:

- detecting risky customers
- reducing false positives
- improving real-world insurance decision-making

Final Selected Threshold:

```python
0.55
```

---

# 📊 Final Model Performance

| Metric | Score |
|---|---|
| Accuracy | 0.6906 |
| Precision | 0.1138 |
| Recall | 0.5448 |
| F1-Score | 0.1883 |
| ROC-AUC | 0.6787 |

---

# 🏆 Final Model

## ✅ Selected Model:
### XGBoost with Class Weights

Why?

- Better minority claim detection
- Stronger business-oriented performance
- Better recall compared to other models
- Stable ROC-AUC performance
- Better balance between risk detection and generalization

---

# 📈 Key Insights

- Accuracy alone is misleading for imbalanced datasets.
- Recall plays a critical role in insurance risk prediction.
- Threshold tuning is essential for business-oriented ML systems.
- XGBoost handled imbalance better than most traditional models.

---

# 🚀 Live Deployment

The Insurance Claim Risk Prediction system has been deployed using Streamlit on Hugging Face Spaces.

🔗 Live Application:  
https://huggingface.co/spaces/gitamacc/insurance-claim-risk-prediction

Users can enter customer and vehicle information to predict insurance claim risk in real time.

---

# 📁 Project Structure

```bash
├── app.py
├── Final_Insurance_Model.pkl
├── Final_Threshold.pkl
├── requirements.txt
├── runtime.txt
├── README.md
```

---

# 👨‍💻 Author

## Surya Teja

Machine Learning & AI Enthusiast  
Focused on building practical business-oriented AI systems.

---

# 📜 License

This project is licensed under the MIT License.
