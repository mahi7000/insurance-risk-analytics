# End-to-End Insurance Risk Analytics & Predictive Modeling

This repository contains the data engineering, exploratory data analysis, and predictive machine learning models built for **AlphaCare Insurance Solutions (ACIS)** to optimize auto-insurance marketing pipelines and implement dynamic, risk-based pricing models in South Africa.

---

## 🚀 Project Overview
ACIS is shifting from intuition-based underwriting to analytics-driven optimization utilizing an 18-month historical auto insurance claims database (Feb 2014 – Aug 2015). 

### Key Performance Indicators (KPIs)
* **Loss Ratio:** $\text{TotalClaims} / \text{TotalPremium}$ (Measures foundational portfolio profitability)
* **Margin:** $\text{TotalPremium} - \text{TotalClaims}$ (Measures absolute monetary contribution per policy)

---

## 📂 Project Directory Structure
```text
insurance-risk-analytics/
├── .github/
│   └── workflows/
│       └── ci.yml             # Automated GitHub Actions CI pipeline
├── data/                      # Tracked by DVC (Data files excluded from Git)
│   └── insurance_data.csv.dvc # Pointer metadata to raw storage
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Profile Assessments
│   ├── 02_hypothesis_testing.ipynb
│   └── 03_modeling.ipynb
├── src/                       # Reusable Object-Oriented production core
│   ├── __init__.py
│   ├── data_loader.py         # Modular pipeline loading routines
│   ├── eda_utils.py
│   ├── hypothesis_tests.py
│   └── modeling.py
├── reports/
│   └── interim_report.md      # Interim status reporting artifact
├── .gitignore
├── requirements.txt           # Standardized dependency tracking
└── README.md