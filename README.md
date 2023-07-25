# Bank Churn Analysis

This repository contains a comprehensive analysis of customer churn in a banking context. The goal of this project is to predict and understand the factors that influence customers' decision to leave the bank, and to explore various techniques to improve model performance.

```Demo Link:``` https://huggingface.co/spaces/yaksh1/Bank-Churn-Analysis

## Dataset Information

The dataset used for this analysis includes the following features:

- **customerID**: Unique identifier for each customer.
- **credit score**: The credit score of the customer.
- **Salary**: The salary of the customer.
- **balance**: The account balance of the customer.
- **satisfaction score**: The satisfaction score of the customer.
- **age**: The age of the customer.
- **tenure**: The number of years the customer has been with the bank.
- **gender**: The gender of the customer (Male or Female).
- **Churn**: Whether the customer churned (1) or not (0).

## Contents

1. **Exploratory Data Analysis (EDA)**:
   - Understanding the distribution of features.
   - Identifying and handling missing values.
   - Exploring relationships between variables.

2. **Data Cleaning**:
   - Handling outliers and anomalies.
   - Feature scaling and normalization.
   - Encoding categorical variables.

3. **Stratified K-fold Cross Validation**:
   - Implementing Stratified K-fold cross-validation to evaluate model performance.
   - Optimizing hyperparameters for better results.

4. **SMOTE-ENN**:
   - Addressing class imbalance using the SMOTE-ENN technique.
   - Balancing the dataset to improve model training.

5. **Machine Learning Models**:
   - Utilizing various machine learning algorithms for churn prediction.
   - Evaluating model performance using metrics like accuracy, precision, recall, and F1-score.

## Getting Started

To get started with this project, clone the repository to your local machine:

```
git clone https://github.com/your-username/bank-churn-analysis.git
```


Feel free to explore the Jupyter Notebook files and the code in this repository. Your feedback and contributions are highly appreciated.
