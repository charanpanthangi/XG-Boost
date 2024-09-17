# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Load dataset
data = pd.read_csv('your_data.csv')

# Univariate Analysis
def univariate_analysis(data):
    print("Univariate Analysis:")
    for column in data.columns:
        plt.figure(figsize=(6,4))
        sns.histplot(data[column], kde=True)
        plt.title(f"Distribution of {column}")
        plt.show()

# Bivariate Analysis (Pair plot for features and target)
def bivariate_analysis(data):
    print("Bivariate Analysis:")
    sns.pairplot(data)
    plt.show()

# Preprocess data
def preprocess_data(data, target_column):
    features = data.drop(columns=[target_column])
    target = data[target_column]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    return train_test_split(X_scaled, target, test_size=0.2, random_state=42)

# Train and evaluate XGBoost Classifier
def train_evaluate_classifier(X_train, X_test, y_train, y_test):
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Classification Accuracy:", accuracy)
    return model

# Train and evaluate XGBoost Regressor
def train_evaluate_regressor(X_train, X_test, y_train, y_test):
    model = XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Regression Mean Squared Error:", mse)
    return model

# Define target columns for classification and regression
classification_target = 'class_target'  # Replace with actual classification target column
regression_target = 'reg_target'        # Replace with actual regression target column

# Preprocess and split data for classification
X_train_cls, X_test_cls, y_train_cls, y_test_cls = preprocess_data(data, classification_target)
# Train and save XGBoost Classifier
xgb_classifier = train_evaluate_classifier(X_train_cls, X_test_cls, y_train_cls, y_test_cls)
with open('xgb_classifier.pkl', 'wb') as file:
    pickle.dump(xgb_classifier, file)

# Preprocess and split data for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = preprocess_data(data, regression_target)
# Train and save XGBoost Regressor
xgb_regressor = train_evaluate_regressor(X_train_reg, X_test_reg, y_train_reg, y_test_reg)
with open('xgb_regressor.pkl', 'wb') as file:
    pickle.dump(xgb_regressor, file)

# Load the saved models and make predictions with new input data
def predict_with_model(model_path, new_input):
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
    new_input_scaled = StandardScaler().fit_transform(new_input)
    prediction = loaded_model.predict(new_input_scaled)
    return prediction

# Example of passing new input data to the classifier
new_input_cls = np.array([[1.5, 2.5, 3.0]])  # Replace with your new input data for classification
print("Classifier Prediction for new input:", predict_with_model('xgb_classifier.pkl', new_input_cls))

# Example of passing new input data to the regressor
new_input_reg = np.array([[1.5, 2.5, 3.0]])  # Replace with your new input data for regression
print("Regressor Prediction for new input:", predict_with_model('xgb_regressor.pkl', new_input_reg))
