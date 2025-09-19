import numpy as np
from flask import Flask, request, render_template
import joblib

app = Flask(__name__,template_folder='Flask/templates',static_folder='Flask/static')

# Load the pre-trained Logistic Regression model and scaler
model = joblib.load('diabetes_logistic_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Predict function to handle POST request from the form
    '''
    # Get input features from the form
    try:
        float_features = [float(x) for x in request.form.values()]
    except ValueError:
        return render_template('index.html', prediction_text="Invalid input! Please provide valid numbers.")

    # Transform features for prediction
    final_features = np.array(float_features).reshape(1, -1)
    scaled_features = scaler.transform(final_features)

    # Get the prediction from the Logistic Regression model
    prediction = model.predict(scaled_features)

    # Interpret the result
    if prediction == 1:
        result = "You have Diabetes. Please consult a doctor."
    else:
        result = "You don't have Diabetes."

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
