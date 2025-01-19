```markdown
# Fraud Detection Using Machine Learning

## Overview

This project implements a **Fraud Detection System** using **Machine Learning** techniques to predict whether a financial transaction is **fraudulent** or **legitimate**. It uses a **Random Forest Classifier** model for classification, with a focus on evaluating the model's performance using various metrics like accuracy, classification report, confusion matrix, and ROC curve.

## Dataset

The dataset used in this project contains **financial transaction data**, including various attributes of the transaction and the customer. The dataset consists of two files:
- **Train Dataset** (`fraudTrain.csv`)
- **Test Dataset** (`fraudTest.csv`)

Each file contains columns related to transaction details, including the target variable `is_fraud` (1 for fraud, 0 for legitimate).

## Features
- **Data Loading**: Loads the train and test datasets from CSV files.
- **Data Preprocessing**: Removes unnecessary columns and applies **One-Hot Encoding** for categorical features.
- **Feature Scaling**: Standardizes the feature set using **StandardScaler** to improve model performance.
- **Model Training**: Trains a **Random Forest Classifier** model to classify fraudulent transactions.
- **Model Evaluation**: Assesses model performance using accuracy, classification report, confusion matrix, and ROC curve.
- **Fraud Prediction**: Allows real-time prediction of fraudulent transactions based on user input.

## Libraries Used
- **Pandas**: For data loading, manipulation, and preprocessing.
- **Numpy**: For numerical operations.
- **Matplotlib** & **Seaborn**: For visualizing the confusion matrix and ROC curve.
- **Scikit-learn**: For machine learning algorithms, model evaluation, and metrics.

## Workflow
1. **Data Loading**: Load the training and test datasets from CSV files.
2. **Data Preprocessing**: Clean the data by removing irrelevant columns and applying One-Hot Encoding to categorical variables.
3. **Feature Scaling**: Standardize the feature set using **StandardScaler**.
4. **Model Training**: Train the **Random Forest Classifier** using the processed data.
5. **Model Evaluation**: Evaluate the performance of the model using accuracy, classification report, confusion matrix, and ROC curve.
6. **Prediction**: Predict if a new transaction is fraudulent or legitimate based on user input.

## Results
- **Random Forest Model**: Achieved high accuracy and strong classification performance in identifying fraudulent transactions.
- **Confusion Matrix**: Visualizes the model's classification performance (True Positives, False Positives, etc.).
- **ROC Curve**: Displays the performance of the model in terms of the True Positive Rate and False Positive Rate, with an area under the curve (AUC) value indicating the model's classification ability.

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

- **Accuracy**: Measures the proportion of correct predictions out of the total predictions.
- **Classification Report**: Includes precision, recall, F1-score for both fraud and legitimate classes.
- **Confusion Matrix**: Provides a breakdown of the model's predictions: True Positives, False Positives, True Negatives, and False Negatives.
- **ROC Curve**: Plots the True Positive Rate against the False Positive Rate to evaluate the model's classification ability.

---

## How to Use the Model

1. Run the script.
2. Enter the transaction details when prompted (e.g., age, income, transaction amount).
3. The model will predict whether the transaction is fraudulent or legitimate based on the input.

```

This README has been corrected and formatted for clarity, including all necessary steps and explanations for using the fraud detection system. Let me know if you'd like further adjustments!
