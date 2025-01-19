import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

train_df = pd.read_csv(r"/Users/sai/Downloads/archive/fraudTrain.csv")
test_df = pd.read_csv(r"/Users/sai/Downloads/archive/fraudTest.csv")


# Combine Train and Test Data
df = pd.concat([train_df, test_df], axis=0)

# Drop Unnecessary Columns
df = df.drop(['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'first', 
              'last', 'street', 'city', 'state', 'zip', 'lat', 'long', 'job', 'dob', 
              'trans_num', 'unix_time', 'merch_lat', 'merch_long'], axis=1)

# One-Hot Encoding for categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Separate Features and Target
X = df_encoded.drop('is_fraud', axis=1)  # Features
y = df_encoded['is_fraud']               # Target

# Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions on the Test Set
rf_pred = rf_model.predict(X_test)

# Evaluation - Random Forest
print("\nRandom Forest Evaluation:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Classification Report:\n", classification_report(y_test, rf_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve - Random Forest
fpr, tpr, thresholds = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label="Random Forest (AUC = {:.3f})".format(roc_auc_score(y_test, rf_pred)))
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Function to predict fraud for a new transaction based on user input
def predict_fraud(transaction_data):
    # Make sure the input data matches the feature columns after encoding
    input_data = pd.DataFrame([transaction_data], columns=X.columns)
    input_data = scaler.transform(input_data)
    prediction = rf_model.predict(input_data)
    
    return "Fraudulent" if prediction[0] == 1 else "Legitimate"

# Collect user input during runtime
def get_user_input():
    # Replace these with actual features from your dataset
    age = float(input("Enter Age: "))
    income = float(input("Enter Income: "))
    transaction_amount = float(input("Enter Transaction Amount: "))
    
    # Add other necessary features here
    # Example: you should include more features from your dataset
    # Make sure to match the features in the same order as your model
    transaction_data = {
        'age': age,
        'income': income,
        'transaction_amount': transaction_amount,
        # Add more features here...
    }
    
    return transaction_data

# Get user input for prediction
user_input = get_user_input()

# Predict fraud for user input
result = predict_fraud(user_input)
print("\nThe transaction is:", result)