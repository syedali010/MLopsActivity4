from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('iris_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)  # Ensure input is in the correct shape
    prediction = model.predict(features)
    # Convert prediction to native Python data type (list of int) before serialization
    prediction_list = prediction.tolist()  # Converts the NumPy array to a Python list
    return jsonify({'prediction': prediction_list})

def home():
    return "Welcome to the Iris Model Flask API!"
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
