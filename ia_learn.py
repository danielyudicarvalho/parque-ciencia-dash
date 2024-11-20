# Import necessary libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix)

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('customer_data.csv')

print('First 5 rows of the dataset')
print(df.head())

print('Dataset information')
print(df.info())

print('Statistical Summary')
print(df.describe())

print('Missing values')
print(df.isnull().sum())

# Fill missing values
df['Income'].fillna(df['Income'].median(), inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

# Define features and target variable
X = df.drop('Target', axis=1)
y = df['Target']

# Split the data into training, validation, and test sets
# First, split off the test set (15% of the data)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42)

# Now, split the remaining data into training and validation sets (85% * 17.65% ≈ 15% of total data)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1765, random_state=42)

# The test_size=0.1765 ensures the validation set is approximately 15% of the total data

# Define models to train
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Support Vector Machine': SVC(kernel='linear', probability=True, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

def train_evaluate(models, X_train, X_val, y_train, y_val):
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        # Predict on the validation set
        y_pred = model.predict(X_val)
        

        # Calculate performance metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

        # Print classification report
        print(f'Performance of {name} on validation set:')
        print(classification_report(y_val, y_pred, zero_division=0))

        # Plot confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

# Train and evaluate models
train_evaluate(models, X_train, X_val, y_train, y_val)

# Assume Random Forest is the best model based on validation performance
best_model = RandomForestClassifier(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)

# Evaluate the best model on the test set
y_test_pred = best_model.predict(X_test)

print('Performance of Best Model on Test Set:')
print(classification_report(y_test, y_test_pred, zero_division=0))

cm_test = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Best Model on Test Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Neural Network Model
nn_model = Sequential()
nn_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
nn_model.add(Dense(8, activation='relu'))
nn_model.add(Dense(1, activation='sigmoid'))

nn_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the neural network with validation data
history = nn_model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_val, y_val), verbose=0)

# Evaluate the neural network on the test set
loss, accuracy = nn_model.evaluate(X_test, y_test, verbose=0)

y_pred_nn = nn_model.predict(X_test)
y_pred_nn = (y_pred_nn > 0.5).astype(int).reshape(-1)

accuracy = accuracy_score(y_test, y_pred_nn)
precision = precision_score(y_test, y_pred_nn, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred_nn, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred_nn, average='weighted', zero_division=0)

print('Performance of Neural Network on Test Set:')
print(classification_report(y_test, y_pred_nn, zero_division=0))
