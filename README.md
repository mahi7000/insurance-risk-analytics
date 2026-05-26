# End-to-End Insurance Risk Analytics & Predictive Modeling

This repository contains the production-grade data engineering, exploratory data analysis, statistical testing, and predictive machine learning models built for **AlphaCare Insurance Solutions (ACIS)** to optimize marketing pipelines and implement dynamic, risk-based premium pricing models.

---

## 🚀 Project Architecture Overview
ACIS is shifting from intuition-based underwriting to data-driven optimization. This project utilizes historical policy and claims data to identify stable, low-risk customer profiles where premiums can be safely optimized to aggressively capture market share while protecting underwriting margins.

### Core Actuarial KPIs
* **Loss Ratio:** $\text{TotalClaims} / \text{TotalPremium}$ (Measures baseline portfolio health)
* **Margin:** $\text{TotalPremium} - \text{TotalClaims}$ (Measures absolute monetary contribution per policy)
* **Pure Risk Premium:** Calculated via a two-stage hurdle architecture to match dynamic consumer risk profiles.

---

## 📂 Project Directory Structure
```text
insurance-risk-analytics/
├── .github/
│   └── workflows/
│       └── ci.yml             # Automated GitHub Actions CI pipeline (Ruff linter)
├── data/                      # Regulated Data Space (Excluded from Git tracking)
│   ├── insurance_data.csv.dvc # DVC Pointer tracking Version 1 (Raw Data)
│   └── cleaned_insurance_data.csv.dvc # DVC Pointer tracking Version 2 (Clean Data)
├── notebooks/
│   ├── 01_eda.ipynb           # Baseline Exploratory Visual Profiling
│   ├── 02_hypothesis_testing.ipynb # Task 3: Regional Statistical A/B Validation
│   └── 03_modeling.ipynb      # Task 4: Two-Stage Machine Learning Pricing Engine
├── src/                       # Reusable Object-Oriented Production Code
│   ├── __init__.py
│   ├── data_loader.py         # Standardized asset ingestion routines
│   ├── hypothesis_tests.py    # Stat engines (Welch's T-Test & Chi-Square)
│   └── modeling.py            # Preprocessing & Dual-Stage Random Forest pipelines
├── reports/
│   └── interim_report.md      # Strategic milestone reporting artifact
├── .gitignore
├── requirements.txt           # Standardized dependency tracking
└── README.md

```

---

## 🛠️ Environment Configuration & Setup

### 1. Initialize Virtual Environment & Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

```

### 2. Resolving Notebook Import Scopes

To enable native workspace utility module imports within local notebook runtime zones without scoping collisions, include the following code block at the head of your execution scripts:

```python
import os
import sys
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

```

---

## 📦 Data Version Control (DVC) Pipeline

This project employs **DVC** to achieve strict, auditable data provenance separate from our source code repository.

### Data Versions Tracked

1. **Version 1 (Raw Data):** `data/insurance_data.csv` — Original historical ledger file.
2. **Version 2 (Cleaned Data):** `data/cleaned_insurance_data.csv` — Filtered state tracking rows with handled null variables and verified feature matrices.

### How to Synchronize Data Assets

If you clone this project onto a fresh environment, pull down the real data objects from your local storage vault remote (`~/dvc_local_vault/acis_storage`) via:

```bash
dvc remote add -d localstorage ~/dvc_local_vault/acis_storage
dvc pull

```

---

## 🧪 Statistical Framework (A/B Testing)

Located in `src/hypothesis_tests.py`, this suite evaluates risk variance across demographic segments and localized Ethiopian geographical regions (e.g., Addis Ababa vs. Oromia):

* **Categorical Risk (Chi-Square Test of Independence):** Validates if claim frequencies scale or differ significantly across distinct driver gender and postal profiles.
* **Numerical Performance (Independent Welch's Two-Sample T-Test):** Evaluates whether mean policy underwriting margins vary with statistical significance across geographic boundaries.

---

## 🤖 Two-Stage Actuarial Pricing Architecture

Auto-insurance data is inherently **zero-inflated** (the majority of policyholders incur zero claim payouts). To prevent severe premium distortions, a **Hurdle/Two-Stage Actuarial Architecture** was engineered in `src/modeling.py`:

1. **Stage 1 (Classification Classifier):** A Random Forest model trains on the total portfolio matrix to calculate an individual's specific probability ($p$) of initiating a claim event.
2. **Stage 2 (Regression Regressor):** A secondary Random Forest model trains exclusively on positive claimant instances to estimate the financial severity ($S$) of a claim event.

### Dynamic Pricing Recommendation Optimization

The ultimate suggested pure baseline premium for any target customer profile is generated programmatically via:


$$\text{Optimized Pure Premium} = p \times S$$

---

## ⚙️ Continuous Integration (CI)

Automated syntax verification checks run seamlessly via **GitHub Actions** on every individual inbound push or Pull Request sequence directed to the `main` branch. The linting rules are executed by the `ruff` execution matrix as defined in `.github/workflows/ci.yml`.