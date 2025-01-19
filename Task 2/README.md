---

# SMS Spam Detection Using Machine Learning

## Overview

This project implements an **SMS Spam Detection System** using **Natural Language Processing (NLP)** and **Machine Learning** techniques. It classifies SMS messages into two categories: **Ham** (non-spam) and **Spam** using two machine learning models: **Naive Bayes** and **Support Vector Machine (SVM)**.

## Dataset

The dataset used in this project is an **SMS Spam Collection** stored in an Excel file. The dataset contains two columns: `v1` (label - "ham" or "spam") and `v2` (SMS message).

## Features
- **Data Loading**: Loads the dataset from an Excel file and processes it into a structured format.
- **Data Preprocessing**: Cleans the text by removing special characters, punctuation, and numbers.
- **Text Vectorization**: Uses **TF-IDF Vectorization** to convert SMS messages into numerical features.
- **Model Training**: Trains two models - **Naive Bayes** and **SVM**.
- **Model Evaluation**: Evaluates both models using accuracy, classification report, and confusion matrix.
- **Prediction**: Classifies new SMS messages as either **Spam** or **Ham**.

## Libraries Used
- **Pandas**: Data handling and preprocessing.
- **Numpy**: Numerical operations.
- **Matplotlib** & **Seaborn**: Visualization of confusion matrices.
- **Scikit-learn**: Machine learning algorithms and evaluation metrics.

## Workflow
1. **Data Loading**: Load dataset from the Excel file.
2. **Data Preprocessing**: Clean the text data (remove numbers, punctuation, convert to lowercase).
3. **Feature Extraction**: Convert text messages into numerical features using **TF-IDF**.
4. **Model Training**: Train two machine learning models: **Naive Bayes** and **SVM**.
5. **Model Evaluation**: Assess performance using accuracy, classification reports, and confusion matrices.
6. **Prediction**: Predict whether an SMS message is spam or ham based on user input.

## Results
- **Naive Bayes Model**: Achieved reliable performance with a good classification score.
- **SVM Model**: Also achieved high performance with clear classification of spam and ham messages.

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
Enter an SMS message (or type 'exit' to stop): Congratulations! You've won a $1000 gift card. Click here to claim it now!
```

**Output:**
```
Naive Bayes Prediction: Spam
SVM Prediction: Spam
```

**Input:**
```
Enter an SMS message (or type 'exit' to stop): Hey, let's catch up tomorrow.
```

**Output:**
```
Naive Bayes Prediction: Ham
SVM Prediction: Ham
```

---

