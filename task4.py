import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATASET_DIR = r'C:\Users\sanik\OneDrive\Desktop\Machine Learning\Task 4\training'  #training folder

# Calorie mapping (random values for now)
calorie_mapping = {
    'Bread': 250,
    'Dairy product': 150,
    'Dessert': 350,
    'Egg': 70,
    'Fried food': 400,
    'Meat': 300,
    'Noodles-Pasta': 350,
    'Rice': 200,
    'Seafood': 200,
    'Soup': 100,
    'Vegetable-Fruit': 50
}

# Prepare the data
X = []  # Features
y = []  # Labels

for label in os.listdir(DATASET_DIR):
    label_dir = os.path.join(DATASET_DIR, label)
    if os.path.isdir(label_dir):
        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            # Read the image and resize it
            img = cv2.imread(img_path)
            img = cv2.resize(img, (64, 64))  # Resize to a smaller size
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            X.append(img.flatten())  # Flatten the image to a 1D array
            y.append(label)  # Append the label

X = np.array(X)
y = np.array(y)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a k-NN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Validate the model
y_pred = knn.predict(X_val)
print(classification_report(y_val, y_pred))
print(f"Accuracy: {accuracy_score(y_val, y_pred)}")

# Save the model
joblib.dump(knn, 'food_recognition_model_knn.pkl')

# Function to predict the class and return calorie content
def predict_calories(img_path):
    # Read and preprocess the image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64))  # Resize to the same size as training
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = img.flatten().reshape(1, -1)  # Flatten and reshape for prediction

    # Predict the class
    predicted_class = knn.predict(img)[0]

    # Get calorie content
    calories = calorie_mapping.get(predicted_class, 0)  # Default to 0 if not found

    return predicted_class, calories

# Prediction
img_path = r"C:\Users\sanik\OneDrive\Desktop\Machine Learning\Task 4\67.jpg"  # image path
predicted_food, calorie_content = predict_calories(img_path)
print(f"Predicted food: {predicted_food}, Estimated Calories: {calorie_content}")
