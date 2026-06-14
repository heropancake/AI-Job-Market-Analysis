# Predictive Labor Market Analysis: AI & Automation Impact

## 🚀 Project Overview
This project delivers a data-driven, end-to-end predictive analytics solution evaluating how Artificial Intelligence (AI) and automation risks influence employee job status across various industries. Utilizing Python and advanced machine learning techniques, the analysis uncovers hidden workforce patterns and models the probability of positions being modified, replaced, or remaining unchanged. 

Designed with a **Business Intelligence and Data Science focus**, this repository serves as a portfolio piece demonstrating data preprocessing, exploratory data analysis (EDA), handling class imbalance, and deploying non-linear classification algorithms.

---

## 🛠️ Tech Stack & Requirements
To replicate or run this environment, ensure you have the following installed:

*   **IDE:** Visual Studio Code (VSC) with official Python and Jupyter extensions.
*   **Core Library:** Python 3.10+
*   **Dependencies:**
```bash
    pip install pandas scikit-learn matplotlib seaborn ipykernel ipympl
    ```
*   *Note on Interactivity:* The exploratory 3D plot leverage `%matplotlib widget` (`ipympl`) to enable dynamic mouse-rotation directly within the notebook interface.

---

## 📊 Dataset & Automated Cloud Pipeline
*   **Target Variable (`Job_Status`):** Classifies roles into *Unchanged*, *Modified*, or *Replaced*.
*   **Key Predictors Included:** `Age`, `Years_Experience`, `Education_Level`, `Industry`, `Remote_Work`, `Upskilling_Required`, `Job_Satisfaction`, and `Salary_Before_AI`.
*   **Cloud Data Loading:** To eliminate local directory path issues (`FileNotFoundError`) and ensure 100% universal reproducibility, the Jupyter Notebook fetches the dataset directly from GitHub's cloud storage using a raw URL pipeline.

---

## 🧠 Machine Learning Methodology & Architecture

### 1. Data Preprocessing & Leakage Prevention
*   **Categorical Encoding:** Text-based variables were transformed into mathematical representations using **One-Hot Encoding** (`pd.get_dummies`).
*   **Feature Scaling:** Continuous numerical inputs (e.g., experience vs. productivity change) were standardized utilizing `StandardScaler`. 
*   *Strict Guardrails:* To avoid catastrophic **Data Leakage**, the feature scaler was strictly fitted *only* on the training split post `train_test_split` (80/20 ratio), preventing validation data from corrupting the model's training matrix.
*   *Feature Engineering Choice:* `Salary_After_AI` was intentionally excluded from the predictors to prevent "time-travel" data leakage, keeping the model practically viable for HR forecasting.

### 2. Evaluated Models
*   **Baseline: Logistic Regression** – Implemented with an increased `max_iter=1000` threshold to ensure algorithmic convergence given the dummy-encoded sparse matrix. Exposed critical class-imbalance limitations regarding the rare `Replaced` category.
*   **Advanced Model 1: Random Forest Classifier** – An ensemble tree-based method configured with `n_estimators=100` to mitigate overfitting and capture non-linear workforce shifts.
*   **Advanced Model 2: K-Nearest Neighbors (KNN)** – A distance-based geometric classifier (`n_neighbors=5`) highly reliant on the previously executed standardization layer.

---

## 📁 Repository Structure
*   `AI_job_market_analysis.ipynb` - The primary production notebook containing documented code cells and analytical logic.
*   `job_market_data.csv` - The source labor dataset used for training and testing.
*   `README.md` - Portfolio documentation and landing page.

---

## ⚠️ Disclaimer & Data Origin
* **Artificial/Synthetic Data:** The dataset used in this project (`job_market_data.csv`) consists entirely of **mock/synthetically generated data**. It does not represent actual employees, real-world corporate structures, or genuine payroll/HR metrics from any specific organization.
* **Educational Purpose Only:** This repository is developed purely for **educational, academic, and portfolio demonstration purposes**. The predictive models, accuracy figures, and analytical insights generated herein should not be used for real-world HR decision-making, workforce planning, or strategic automation forecasting.