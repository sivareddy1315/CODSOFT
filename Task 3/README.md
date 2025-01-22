

---

# Credit Card Fraud Detection Using Machine Learning

## Overview

This project implements a **Credit Card Fraud Detection System** using **Machine Learning** techniques. The system detects fraudulent transactions based on user behavior and transaction details using a **Random Forest Classifier**.

## Dataset

The dataset contains details about credit card transactions, including attributes such as transaction amount, location, time, and user demographics. The dataset is split into training and testing sets to evaluate the model.

## Features

- **Data Preprocessing**: Combines training and testing datasets, removes unnecessary features, and handles categorical data through one-hot encoding.
- **Feature Scaling**: Standardizes the data for optimal model performance using `StandardScaler`.
- **Machine Learning Models**: Implements the **Random Forest Classifier** to predict fraudulent transactions.
- **Evaluation Metrics**: Evaluates model performance using accuracy, classification reports, confusion matrices, and ROC curves.
- **Fraud Prediction**: Accepts real-time user inputs to predict if a transaction is fraudulent or legitimate.

## Libraries Used

- **Pandas**: Data manipulation and preprocessing.
- **NumPy**: Numerical computation.
- **Scikit-learn**: Machine learning algorithms and evaluation metrics.
- **Matplotlib & Seaborn**: Visualization of results and performance metrics.

## Workflow

1. **Data Loading**: Combines training and testing datasets into a single DataFrame.
2. **Data Cleaning**: Drops irrelevant columns and handles categorical variables using one-hot encoding.
3. **Feature Scaling**: Normalizes numeric features using **StandardScaler** to improve model performance.
4. **Model Training**: Trains a **Random Forest Classifier** using the preprocessed data.
5. **Evaluation**: Evaluates the model performance using confusion matrices, ROC curves, and accuracy scores.
6. **Prediction**: Accepts real-time user input to predict whether a transaction is fraudulent or legitimate.

## Results

- **Random Forest Model**: The model achieved high accuracy and effectively identified fraudulent transactions.
- **ROC Curve Analysis**: The model demonstrated excellent performance in distinguishing between legitimate and fraudulent transactions.

## Requirements

- **Python 3.x**
- **Required Libraries**:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn

Install the required libraries using:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Example Input and Output

**Example Input**:

```
Enter Age: 30
Enter Income: 50000
Enter Transaction Amount: 1200
```

**Example Output**:

```
The transaction is: Fraudulent
```

---

