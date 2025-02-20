import os
import numpy as np
import requests
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('model/model (2).keras')
print('Model loaded. Check http://127.0.0.1:5000/')

# Disease labels
labels = {0: 'Healthy', 1: 'Powdery Mildew', 2: 'Rust'}

# Weather API function
def get_weather(city, state):
    api_key = "39412fa6d262550086ce698f977c3530"  # Replace with your OpenWeather API key
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},{state},IN&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    if response.status_code == 200:
        return data['main']['temp'], data['main']['humidity']
    return None, None

# Image Prediction function
def get_result(image_path):
    try:
        img = load_img(image_path, target_size=(225, 225))
        x = img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        predictions = model.predict(x)[0]
        return predictions
    except Exception as e:
        return str(e)

# Get Recommendations
def get_recommendations(disease):
    recommendations = {
        "Healthy": {
            "preventive": "Maintain proper watering and nutrient supply.",
            "treatment": "No treatment needed. Keep monitoring your crops."
        },
        "Powdery Mildew": {
            "preventive": "Ensure good air circulation, avoid overhead watering.",
            "treatment": "Use sulfur-based fungicides. Neem oil can also help."
        },
        "Rust": {
            "preventive": "Rotate crops, remove infected leaves.",
            "treatment": "Apply fungicides like copper-based sprays."
        }
    }
    return recommendations.get(disease, {})

# Home Route
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files or 'city' not in request.form or 'state' not in request.form:
        return jsonify({'error': 'Missing file, city, or state information'})
    
    file = request.files['file']
    city = request.form['city']
    state = request.form['state']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Save uploaded image
        upload_path = os.path.join(os.path.dirname(__file__), 'uploads')
        os.makedirs(upload_path, exist_ok=True)  # Ensure 'uploads/' directory exists
        file_path = os.path.join(upload_path, secure_filename(file.filename))
        file.save(file_path)

        # Get weather data
        temperature, humidity = get_weather(city, state)
        if temperature is None:
            return jsonify({'error': 'Unable to fetch weather data'})

        # Predict disease
        predictions = get_result(file_path)
        if isinstance(predictions, str):
            return jsonify({'error': predictions})  # If error occurs in prediction

        predicted_label = labels[np.argmax(predictions)]
        recommendations = get_recommendations(predicted_label)

        return jsonify({
            'disease': predicted_label,
            'temperature': temperature,
            'humidity': humidity,
            'preventive_measures': recommendations.get('preventive', 'No data available'),
            'treatment': recommendations.get('treatment', 'No data available')
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
