# 📊 HR Salary Predictor — XGBoost + SHAP Explainability

**Author:** Tamjeed Rehman — AI Developer & Data Scientist  
**Contact:** tamjeedrehman1@gmail.com

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RAnoWefBxl1JK2e-MROUIIHQxwjiuq_p?usp=sharing)

---

## Project Overview

This project builds a **professional HR salary prediction system** using 1,000 real employee records. HR agencies and recruitment companies use exactly this type of model to determine fair compensation, detect pay gaps, and make data-driven hiring decisions.

> **Business Value:** HR agencies pay $200–$500 for this exact solution. It eliminates guesswork from compensation decisions and ensures fair, data-driven salary offers.

---

## What This Project Demonstrates

- ✅ Real-world HR data cleaning & exploratory data analysis
- ✅ Advanced feature engineering (experience bins, interaction features)
- ✅ Training & comparing 3 ML models with 5-fold cross-validation
- ✅ Hyperparameter tuning with GridSearchCV
- ✅ SHAP values for explainable salary predictions
- ✅ Salary fairness analysis (gender & location pay gaps)
- ✅ Interactive salary calculator

---

## Dataset

| Property | Detail |
|----------|--------|
| Source | Kaggle — HR Salary Prediction Dataset |
| Records | 1,000 real employee records |
| Features | Education, Experience, Location, Job Title, Age, Gender |
| Target | Annual Salary (USD) |

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation & DataFrames |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Professional visualizations |
| `scikit-learn` | ML models, cross-validation, encoding |
| `xgboost` | Best performing model |
| `shap` | Explainable AI — why did the model predict this salary? |

---

## Project Steps

### Step 1 — Setup & Data Loading
Install libraries, upload dataset, load 1,000 HR employee records.

### Step 2 — Exploratory Data Analysis
- Salary distribution by role, education, gender, location
- Key finding: PhD holders earn avg **$136K** vs High School avg **$77K**
- Urban workers earn **$110.9K** vs Rural **$98.7K**

### Step 3 — Data Visualization
6 professional charts:
1. Salary by Job Title
2. Salary by Education Level
3. Experience vs Salary scatter
4. Gender Pay Gap by role
5. Salary by Location
6. Overall Salary Distribution

### Step 4 — Feature Engineering
12 predictive features engineered:

| Feature | Type | Why It Matters |
|---------|------|----------------|
| `Education_Enc` | Label encoded | ML needs numbers, not text |
| `Job_Title_Enc` | Label encoded | Captures role hierarchy |
| `Location_Enc` | Label encoded | Geographic salary premium |
| `Gender_Enc` | Binary encoded | For fairness analysis |
| `Exp_Level` | Binned | Junior/Mid/Senior/Lead tier |
| `Exp_Squared` | Polynomial | Captures diminishing returns |
| `Age_Group` | Binned | Career stage grouping |
| `Role_Exp` | Interaction | Role × experience combined signal |

### Step 5 — Model Training & Evaluation
3 models compared with 5-fold cross-validation:

| Model | R² (Test) | R² (CV) | RMSE ($) | MAE ($) |
|-------|-----------|---------|----------|---------|
| Linear Regression | 0.5679 | 0.5414 | 18,783 | 15,393 |
| Random Forest | 0.8434 | 0.8343 | 11,309 | 9,260 |
| **XGBoost (tuned)** | **0.8631** | **0.8629** | **10,573** | **8,566** |

✅ **Best Model: XGBoost with GridSearchCV tuning**

### Step 6 — SHAP Explainability + Fairness Analysis
- SHAP feature importance: Education is the #1 salary driver
- Overall gender pay gap: **+1.5%** (male earns more)
- Location premium: Urban **$110.9K** > Suburban **$107.6K** > Rural **$98.7K**

### Step 7 — Interactive Salary Calculator
Enter any employee profile → get instant salary prediction with confidence range.

**Sample predictions:**
```
Director | PhD    | 15yr | Urban    | Male   →  $150,991  ($135,892–$166,090)
Analyst  | Bachelor | 3yr | Suburban | Female →  $57,392   ($51,653–$63,131)
Manager  | Master | 8yr  | Urban    | Male   →  $118,819  ($106,937–$130,701)
```

---

## Results Summary

| Metric | Result |
|--------|--------|
| Dataset | 1,000 real HR employee records |
| Features Engineered | 12 predictive features |
| Models Trained | 3 (Linear Regression, Random Forest, XGBoost) |
| Validation | 5-fold cross-validation |
| Best Model | XGBoost with GridSearchCV tuning |
| Best R² Score | **0.8631** |
| Best MAE | **$8,566** |
| Explainability | SHAP values for every prediction |
| Fairness Analysis | Gender pay gap by role and location |

---

## Skills Demonstrated

`Data Cleaning` · `Exploratory Data Analysis` · `Feature Engineering` · `Label Encoding` · `Interaction Features` · `Cross-Validation` · `GridSearchCV` · `XGBoost` · `SHAP` · `Fairness Analysis` · `Python` · `Scikit-learn`

---

## How to Run

1. Click the **Open in Colab** badge above
2. Upload the dataset zip file when prompted (`archive_2_.zip`)
3. Run all cells in order

---

*Built by Tamjeed Rehman — AI Developer & Data Scientist | tamjeedrehman1@gmail.com*
