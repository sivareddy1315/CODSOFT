```yaml
---
# SMS Spam Detection Using Machine Learning

## Overview

This project implements an **SMS Spam Detection System** using **Natural Language Processing (NLP)** and **Machine Learning** techniques. It classifies SMS messages into two categories: **Ham** (non-spam) and **Spam** using two machine learning models: **Naive Bayes** and **Support Vector Machine (SVM)**.

## Dataset

The dataset used in this project is an **SMS Spam Collection** stored in an Excel file. The dataset contains two columns:
- `v1` (label - "ham" or "spam")
- `v2` (SMS message)

This dataset is pre-processed for text cleaning and transformed for training machine learning models.

## Features

- **Data Loading**: Loads the dataset from an Excel file and processes it into a structured format.
- **Data Preprocessing**: Cleans the text by removing special characters, punctuation, and numbers, and converts text to lowercase.
- **Text Vectorization**: Uses **TF-IDF Vectorization** to convert SMS messages into numerical features.
- **Model Training**: Trains two models - **Naive Bayes** and **SVM**.
- **Model Evaluation**: Evaluates both models using accuracy, classification report, and confusion matrix.
- **Prediction**: Classifies new SMS messages as either **Spam** or **Ham**.

## Libraries Used

- **Pandas**: For data handling and preprocessing.
- **Numpy**: For numerical operations.
- **Matplotlib** & **Seaborn**: For visualizing confusion matrices and other evaluation metrics.
- **Scikit-learn**: For machine learning algorithms, text vectorization, and evaluation metrics.

## Workflow

1. **Data Loading**: Load the SMS Spam dataset from the Excel file.
2. **Data Preprocessing**: Clean the text data (remove numbers, punctuation, and convert to lowercase).
3. **Feature Extraction**: Convert SMS messages into numerical features using **TF-IDF** Vectorization.
4. **Model Training**: Train two machine learning models: **Naive Bayes** and **SVM**.
5. **Model Evaluation**: Evaluate both models using accuracy, classification reports, and confusion matrices.
6. **Prediction**: Classify new SMS messages as either spam or ham based on user input.

## Results

- **Naive Bayes Model**: Achieved reliable performance with a good classification score and efficiency.
- **SVM Model**: Also achieved high performance with clear classification of spam and ham messages, and worked well with the dataset.

## Requirements

- Python 3.x
- Required Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

To install the necessary libraries, you can use the following command:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Example Input and Output

### Input 1:
```
Enter an SMS message (or type 'exit' to stop): Congratulations! You've won a $1000 gift card. Click here to claim it now!
```

### Output 1:
```
Naive Bayes Prediction: Spam
SVM Prediction: Spam
```

### Input 2:
```
Enter an SMS message (or type 'exit' to stop): Hey, let's catch up tomorrow.
```

### Output 2:
```
Naive Bayes Prediction: Ham
SVM Prediction: Ham
```

## Conclusion

This project successfully implements an SMS Spam Detection system using machine learning techniques. It uses **Naive Bayes** and **SVM** for classification, achieving good results for detecting spam messages. The system can be further improved by using additional techniques like deep learning or fine-tuning hyperparameters for even better performance.
```

This format should be compatible for use in a Git repository, with proper markdown structure. You can copy this directly to a `README.md` file and it will display correctly in GitHub or other platforms that support markdown.
