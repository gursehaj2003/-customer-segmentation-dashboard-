# 🛒 Customer Segmentation & Churn Prediction Project

An end-to-end **Customer Analytics** project that performs **customer segmentation using K-Means clustering**, **churn prediction using Random Forest**, and visualizes insights through an **interactive Streamlit dashboard** and **Power BI**.

This project simulates a real-world retail analytics pipeline — from raw data to ML models to business dashboards.

---

## 🚀 Project Overview

### Objectives
- Segment customers based on purchasing behavior
- Predict customer churn using machine learning
- Visualize insights via Streamlit & Power BI
- Build a reproducible ML pipeline suitable for real-world deployment

---

## 📊 Dataset

- **Source:** Kaggle  
- **File:** `online_retail_customer_data_extended.csv`
- **Description:**  
  Contains customer demographics, purchasing behavior, satisfaction scores, and churn labels.

---

## 🧠 Machine Learning Techniques Used

### Customer Segmentation
- Algorithm: **K-Means Clustering**
- Features:
  - Age
  - Annual Income
  - Spending Score
  - Total Purchases
  - Average Purchase Value
  - Satisfaction Score

### Churn Prediction
- Algorithm: **Random Forest Classifier**
- Features:
  - Behavioral features
  - Demographics
  - Website visit frequency
- Target:
  - `Churn`

---

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **Libraries:**
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - plotly
  - streamlit
  - joblib
- **Tools:**
  - Anaconda
  - Power BI

---

## 📁 Project Structure


---

## ⚙️ Environment Setup

### Step 1: Create Conda Environment
```bash
conda create -n customer_seg python=3.11 -y
conda activate customer_seg

🔄 Pipeline Execution
python prep_data.py
python segmentation.py
python churn_model.py
streamlit run app.py
working output link http://localhost:8501



✅ Key Outputs

Customer Segmentation using K-Means

Churn Prediction using Random Forest

🙌 Author

Gursehaj Singh
Customer Segmentation & Churn Analytics Project
