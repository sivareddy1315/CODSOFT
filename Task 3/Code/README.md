```markdown
# Fraud Detection Using Machine Learning

## Overview

This project implements a **Fraud Detection System** using **Machine Learning** techniques to predict whether a financial transaction is **fraudulent** or **legitimate**. It uses two machine learning models: **Random Forest Classifier** and **Logistic Regression**, with a focus on evaluating and comparing the performance of these models.

## Dataset

The dataset used in this project contains **financial transaction data**, including various attributes of the transaction and the customer. The dataset has two files:
- **Train Dataset** (`fraudTrain.csv`)
- **Test Dataset** (`fraudTest.csv`)

Each file contains columns related to transaction details, including the target variable `is_fraud` (1 for fraud, 0 for legitimate).

## Features
- **Data Loading**: Loads the train and test datasets from CSV files.
- **Data Preprocessing**: Drops unnecessary columns and applies **One-Hot Encoding** for categorical features.
- **Feature Scaling**: Standardizes the feature set using **StandardScaler**.
- **Model Training**: Trains a **Random Forest Classifier** model to classify fraudulent transactions.
- **Model Evaluation**: Evaluates the model performance using accuracy, classification report, confusion matrix, and ROC curve.
- **Fraud Prediction**: Allows real-time prediction of fraudulent transactions based on user input.

## Libraries Used
- **Pandas**: Data loading and preprocessing.
- **Numpy**: Numerical operations.
- **Matplotlib** & **Seaborn**: Visualization of confusion matrix and ROC curve.
- **Scikit-learn**: Machine learning algorithms and evaluation metrics.

## Workflow
1. **Data Loading**: Load the training and test datasets from CSV files.
2. **Data Preprocessing**: Clean the data by removing irrelevant columns and encoding categorical variables.
3. **Feature Scaling**: Apply standardization to numerical features.
4. **Model Training**: Train a **Random Forest Classifier** model using the processed data.
5. **Model Evaluation**: Evaluate the model performance using accuracy, classification report, confusion matrix, and ROC curve.
6. **Prediction**: Predict if a new transaction is fraudulent or legitimate based on user input.

## Results
- **Random Forest Model**: Achieved high accuracy and strong classification performance for identifying fraudulent transactions.
- **Confusion Matrix**: Visualizes the model's classification performance (True Positives, False Positives, etc.).
- **ROC Curve**: Displays the performance of the model in terms of True Positive Rate and False Positive Rate.

## Requirements
- Python 3.x
- Required Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

You can install the required libraries using the following command:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Example Input and Output

**Input:**
```
Enter Age: 34
Enter Income: 45000
Enter Transaction Amount: 200
```

**Output:**
```
The transaction is: Legitimate
```

---

## Model Evaluation Metrics

- **Accuracy**: Measures the proportion of correct predictions.
- **Classification Report**: Includes precision, recall, F1-score for both classes (fraudulent and legitimate).
- **Confusion Matrix**: Provides a detailed breakdown of the model's predictions against actual labels.
- **ROC Curve**: Measures the trade-off between True Positive Rate and False Positive Rate.

---

## How to Use the Model

1. Run the script.
2. Enter the transaction details when prompted (e.g., age, income, transaction amount).
3. The model will predict whether the transaction is fraudulent or legitimate.

---

```

