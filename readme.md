# DA5401 Assignment 6 – Imputation via Regression for Missing Data  
**Pragati L - CE22B089**  

---

## Overview
This notebook explores strategies for handling missing data in the **UCI Credit Card Default Clients Dataset** and evaluates the impact of these strategies on a downstream classification task. The main focus is on:

- **Simple Imputation (Median)** – Filling missing values with the median of each column.  
- **Regression Imputation (Linear and Non-Linear)** – Predicting missing values using linear regression or non-linear methods (KNN / Decision Tree).  
- **Listwise Deletion** – Removing all rows containing missing values.  

The goal is to understand how missing data affects classifier performance and why selecting an appropriate imputation method is critical for real-world datasets.

---

## Dataset
**Source:** [UCI Credit Card Default Clients Dataset on Kaggle](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset)  
**Features:** Various demographic, billing, and payment attributes (numerical).  
**Target:** `default payment next month` (binary classification).  
**Preprocessing:**  
- Artificially introduce Missing At Random (MAR) values (5–10%) in selected columns such as `AGE`, `BILL_AMT1`, `PAY_AMT1` to simulate real-world missing data scenarios.  

---

## Notebook Structure

### Part A: Data Preprocessing and Imputation
1. **Load and Prepare Data**  
   - Load the dataset and inspect features and target.  
   - Introduce MAR missing values in selected numerical columns.

2. **Imputation Strategy 1: Simple Imputation (Baseline)**  
   - Create Dataset A by filling missing values with the **median**.  
   - Discuss why median is preferred over mean.

3. **Imputation Strategy 2: Regression Imputation (Linear)**  
   - Create Dataset B by predicting missing values in one selected column using **Linear Regression**.  
   - Discuss the **Missing At Random (MAR)** assumption.

4. **Imputation Strategy 3: Regression Imputation (Non-Linear)**  
   - Create Dataset C by predicting missing values using **non-linear regression models** (KNN or Decision Tree).  

---

### Part B: Model Training and Performance Assessment
1. **Data Split**  
   - Split Datasets A, B, C into training and testing sets.  
   - Create Dataset D using **Listwise Deletion** and split.  

2. **Feature Scaling**  
   - Standardize all datasets using **StandardScaler** to normalize feature magnitudes.

3. **Classifier Setup and Evaluation**  
   - Train a **Logistic Regression** classifier on each dataset.  
   - Evaluate models using **accuracy, precision, recall, and F1-score**.  

---

### Part C: Comparative Analysis
1. **Results Comparison**  
   - Summarize classification performance across all datasets in a table, focusing on F1-scores.  

2. **Efficacy Discussion**  
   - Discuss the trade-offs between **Listwise Deletion** and **imputation strategies**.  
   - Analyze which regression method (Linear vs Non-Linear) performed better and why.  
   - Recommend the **best strategy** for handling missing data based on both empirical performance and conceptual reasoning.  

---

## How to Run
1. Install dependencies:  
pip install numpy pandas matplotlib scikit-learn

2. Download the dataset from the link : https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset and change the path of the file accordingly.

3. Run all the cells sequentially.
