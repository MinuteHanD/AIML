import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np
import joblib

# Load the dataset
dataset = pd.read_csv('merged.csv')  # Replace 'dataset.csv' with the actual filename/path

# Drop rows with NaN values in the 'text' column
dataset.dropna(subset=['text'], inplace=True)

# Split the data into features and labels
X = dataset['text']
y = dataset['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer to convert text into numerical features
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train.astype('U'))  # Convert X_train to Unicode strings
X_test_vec = vectorizer.transform(X_test.astype('U'))  # Convert X_test to Unicode strings

# Train an SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_vec, y_train)

# Save the trained model
joblib.dump(svm_model, 'trained_model.pkl') 

# Make predictions on the test set
y_pred = svm_model.predict(X_test_vec)

# Evaluate the model
print(classification_report(y_test, y_pred))
