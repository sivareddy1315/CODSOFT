# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset from Excel File
df = pd.read_excel(r"/Users/sai/CODSOFT/Task 2/Dataset/spam.xlsx")  # Update path if needed

# Drop Unnecessary Columns
df = df[['v1', 'v2']]  # Keep only label and message columns
df.columns = ['label', 'message']  # Rename columns for clarity

# Label Encoding: Convert 'ham' to 0 and 'spam' to 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Handle Missing or Non-String Values in 'message'
df['message'] = df['message'].astype(str)  # Ensure all values are strings

# Preprocess the Text Data
import re
import string

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()  # Remove extra spaces
    return text

df['message'] = df['message'].apply(clean_text)

# Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['message']).toarray()
y = df['label']

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Model 1: Naive Bayes ---
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

print("\nNaive Bayes Evaluation:")
print("Accuracy:", accuracy_score(y_test, nb_pred))
print("Classification Report:\n", classification_report(y_test, nb_pred))

# Confusion Matrix for Naive Bayes
cm_nb = confusion_matrix(y_test, nb_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.title('Confusion Matrix - Naive Bayes')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# --- Model 2: Support Vector Machine (SVM) ---
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

print("\nSVM Evaluation:")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print("Classification Report:\n", classification_report(y_test, svm_pred))

# Confusion Matrix for SVM
cm_svm = confusion_matrix(y_test, svm_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.title('Confusion Matrix - SVM')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# --- Prediction for New SMS Messages (Runtime Input) ---
while True:
    # Get user input for SMS message
    message = input("\nEnter an SMS message (or type 'exit' to stop): ")
    if message.lower() == 'exit':  # Exit loop if user types 'exit'
        break

    # Clean the input message
    cleaned_message = clean_text(message)

    # Convert input message to TF-IDF features
    input_features = vectorizer.transform([cleaned_message]).toarray()

    # Predict using both models
    nb_prediction = nb_model.predict(input_features)[0]
    svm_prediction = svm_model.predict(input_features)[0]

    # Display predictions
    print(f"Naive Bayes Prediction: {'Spam' if nb_prediction == 1 else 'Ham'}")
    print(f"SVM Prediction: {'Spam' if svm_prediction == 1 else 'Ham'}")