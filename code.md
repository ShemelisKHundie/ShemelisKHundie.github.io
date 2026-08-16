---
layout: default
title: Code & Reproducibility
---

# Code & Reproducibility

This page documents all Stata and Python scripts used in my research, with clear links to the papers they support. Each entry includes a description of the workflow, the empirical strategy, and the corresponding publication or working paper.

---

## 📘 Stata Workflows
### **Paper**
1. Hundie, S. K., & Csapi, V. (2026). *Nonlinear effects of economic growth on environmental sustainability in Hungary: the roles of globalization and energy imports.*Energy, Ecology and Environment, 1–26. https://doi.org/10.1007/s40974-026-00409-8
### **Data**
(Will be attached)

### **Code**
`ardl_nardl_hungary.do`

**Description:**  
This Stata script estimates linear and nonlinear ARDL models to examine long-run and short-run dynamics between globalization, energy use, and environmental outcomes. Includes unit root testing, lag selection, bounds testing, and dynamic multipliers.

---

### 2. Impact Evaluation (DID / IV / FE)  
**Paper:** *Impact of Financial Inclusion on Household Welfare*  
**Code:** [`impact_eval.do`](code/stata/impact_eval.do)  
**Description:**  
Implements difference-in-differences, instrumental variables, and fixed-effects models. Includes data cleaning, treatment assignment, parallel trends diagnostics, and robustness checks.

---

### 3. Spatial Durbin Model (SDM)  
**Paper:** *Spatial Effects of ESG Performance Across EU Regions*  
**Code:** [`sdm_esg.do`](code/stata/sdm_esg.do)  
**Description:**  
Runs spatial lag, spatial error, and SDM models using spatial weight matrices. Includes Moran’s I tests, LM diagnostics, and impact decomposition.

---

## 🐍 Python Workflows

### 4. Topic Modeling (NLP)  
**Paper:** *Text-Based ESG Disclosure Analysis*  
**Code:** [`esg_topic_modeling.ipynb`](code/python/esg_topic_modeling.ipynb)  
**Description:**  
Python notebook performing preprocessing, TF–IDF, LDA topic modeling, coherence scoring, and visualization of ESG disclosure themes.

---

### 5. Dynamic Simulations  
**Paper:** *Energy–Environment–Economy Dynamic Interaction Model*  
**Code:** [`dynamic_simulation.py`](code/python/dynamic_simulation.py)  
**Description:**  
Simulates nonlinear interactions between energy consumption, emissions, and economic growth using differential equations and Monte Carlo sensitivity analysis.

---

### 6. Machine Learning for ESG Prediction  
**Paper:** *Predicting ESG Scores Using Financial and Textual Data*  
**Code:** [`esg_ml_pipeline.ipynb`](code/python/esg_ml_pipeline.ipynb)  
**Description:**  
End-to-end ML pipeline including feature engineering, model training (RF, XGBoost), cross-validation, SHAP interpretability, and out-of-sample forecasting.

---

## 🔗 Replication Packages

### Full Replication Repositories  
- **ESG Nexus Paper:** https://github.com/ShemelisKHundie/esg-nexus-replication  
- **Financial Inclusion Impact Evaluation:** https://github.com/ShemelisKHundie/fin-inclusion-impact  
- **ESG Text Analysis:** https://github.com/ShemelisKHundie/esg-text-analysis  

Each repository contains data, code, and documentation for full reproducibility.

---

# 📌 Notes  
- All scripts follow reproducible workflow standards.  
- Code is organized by paper and method.  
- Updates will be added as new papers are published.
