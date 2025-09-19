import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load dataset
dataset = pd.read_csv('diabetes.csv')

# Preprocessing: select features and target
X = dataset.iloc[:, [1, 4, 5, 7]].values  # Features: Glucose, Insulin, BMI, Age
y = dataset.iloc[:, 8].values  # Target: Outcome

# Scale features
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=16)

# Train the Logistic Regression model
model = LogisticRegression(random_state=16)
model.fit(X_train, y_train)

# Save the model and scaler using joblib
joblib.dump(model, 'diabetes_logistic_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

# Optional: print model accuracy
accuracy = model.score(X_test, y_test)
print(f'Model Accuracy: {accuracy * 100:.2f}%')
