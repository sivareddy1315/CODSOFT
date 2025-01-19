import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Function to load data
def load_data(file_path):
    try:
        # Load the dataset with appropriate column names
        data = pd.read_csv(file_path, sep=':::', names=['Title', 'Genre', 'Description'], engine='python')
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Function to clean text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text.lower()  # Convert to lowercase

# File path to the dataset
file_path = "/Users/sai/CODSOFT/Task 1/Dataset/train_data.txt"

# Load and preprocess the data
data = load_data(file_path)
if data is not None:
    print("Sample data loaded successfully:")
    print(data.head())

    # Drop missing values
    data.dropna(subset=['Description', 'Genre'], inplace=True)

    # Normalize the titles and clean descriptions
    data['Title'] = data['Title'].str.strip().str.lower()
    data['clean_description'] = data['Description'].apply(preprocess_text)

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(data['clean_description'])
    y = data['Genre']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)

    # Evaluate the model
    y_pred = classifier.predict(X_test)
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_).plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

    # Predict genre for a movie based on its title
    while True:
        # Prompt the user to enter a movie title
        movie_title = input("\nEnter the movie title to predict its genre (or type 'quit' to exit): ").strip().lower()
        
        if movie_title == 'quit':
            print("Exiting the program. Goodbye!")
            break

        # Search for the movie in the dataset
        movie_data = data[data['Title'] == movie_title]
        
        if not movie_data.empty:
            # Get the movie description
            description = movie_data['clean_description'].values[0]
            
            # Transform the description using the trained TF-IDF vectorizer
            description_vector = vectorizer.transform([description])
            
            # Predict the genre
            predicted_genre = classifier.predict(description_vector)[0]
            print(f"\nMovie: {movie_title.title()}")
            print(f"Description: {movie_data['Description'].values[0]}")
            print(f"Predicted Genre: {predicted_genre}")
        else:
            print(f"\nMovie '{movie_title}' not found in the dataset. Please try another title.")

else:
    print("Data could not be loaded. Please check the file path and format.")
