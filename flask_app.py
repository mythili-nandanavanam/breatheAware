from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()
import joblib
import numpy as np
import requests
import os
from datetime import datetime

api_key = os.getenv("OPENWEATHER_API_KEY")
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load pre-trained ML models at startup
try:
    model = joblib.load('aqi_classification_model_random_forest.joblib')
    label_encoder = joblib.load('aqi_label_encoder.joblib')
    feature_names = joblib.load('feature_names.joblib')
    print("✅ ML models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model = None
    label_encoder = None
    feature_names = None

# AQI class mappings with emojis and colors
AQI_INFO = {
    'Good': {
        'emoji': '🟢',
        'color': '#00E400',
        'range': '0-50',
        'health_tip': 'Air quality is excellent! Perfect for outdoor activities and exercise. 🏃‍♀️'
    },
    'Moderate': {
        'emoji': '🟡',
        'color': '#FFFF00',
        'range': '51-100',
        'health_tip': 'Air quality is acceptable. Sensitive individuals should consider limiting prolonged outdoor exertion. 🚶‍♂️'
    },
    'Unhealthy for Sensitive Groups': {
        'emoji': '🟠',
        'color': '#FF7E00',
        'range': '101-150',
        'health_tip': 'Sensitive groups should reduce outdoor activities. Children and elderly should stay indoors. 🏠'
    },
    'Unhealthy': {
        'emoji': '🔴',
        'color': '#FF0000',
        'range': '151-200',
        'health_tip': 'Everyone should limit outdoor activities. Wear masks when going outside. 😷'
    },
    'Very Unhealthy': {
        'emoji': '🟣',
        'color': '#8F3F97',
        'range': '201-300',
        'health_tip': 'Health alert! Avoid outdoor activities. Keep windows closed and use air purifiers. ⚠️'
    },
    'Hazardous': {
        'emoji': '🟤',
        'color': '#7E0023',
        'range': '301+',
        'health_tip': 'Emergency conditions! Stay indoors at all times. Seek medical attention if needed. 🚨'
    }
}

def get_aqi_value_from_class(aqi_class):
    """Convert AQI class to approximate numerical value using midpoint"""
    midpoints = {
        'Good': 25,
        'Moderate': 75,
        'Unhealthy for Sensitive Groups': 125,
        'Unhealthy': 175,
        'Very Unhealthy': 250,
        'Hazardous': 350
    }
    return midpoints.get(aqi_class, 100)

@app.route('/')
def home():
    return jsonify({
        'message': 'BreatheAware API is running! 🌍',
        'endpoints': {
            '/predict': 'POST - Predict AQI from pollutant values',
            '/live-aqi': 'GET - Get live AQI for Hyderabad'
        }
    })

@app.route('/predict', methods=['POST'])
def predict_aqi():
    try:
        if not model or not label_encoder or not feature_names:
            return jsonify({'error': 'ML models not loaded properly'}), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        # Expected pollutant values
        required_fields = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']
        
        # Validate input
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        print("📩 Received JSON:", data)

        # Extract pollutant values
        pollutants = [
            float(data['pm25']),
            float(data['pm10']),
            float(data['no2']),
            float(data['so2']),
            float(data['co']),
            float(data['o3'])
        ]
        
        # Create feature array
        features = np.array([pollutants])
        
        # Make prediction
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]
        
        # Get class name from label encoder
        aqi_class = label_encoder.inverse_transform([prediction])[0]
        
        # Get confidence (probability of predicted class)
        confidence = round(max(model.predict_proba([features])[0]) * 100, 1)
        
        # Get AQI numerical value
        aqi_value = get_aqi_value_from_class(aqi_class)
        
        # Get additional info
        aqi_info = AQI_INFO.get(aqi_class, AQI_INFO['Moderate'])
        
        return jsonify({
            'success': True,
            'aqi_class': aqi_class,
            'aqi_value': aqi_value,
            'confidence': round(confidence, 2),
            'emoji': aqi_info['emoji'],
            'color': aqi_info['color'],
            'range': aqi_info['range'],
            'health_tip': aqi_info['health_tip'],
            'pollutants': {
                'pm25': pollutants[0],
                'pm10': pollutants[1],
                'no2': pollutants[2],
                'so2': pollutants[3],
                'co': pollutants[4],
                'o3': pollutants[5]
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/live-aqi', methods=['GET'])
def get_live_aqi():
    """Endpoint to get live AQI data for Hyderabad"""
    try:
        api_key = os.getenv("OPENWEATHER_API_KEY")
        if not api_key:
            return jsonify({'error': 'OpenWeatherMap API key missing'}), 500
        
        # Hyderabad coordinates
        lat, lon = 17.385, 78.4867
        url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
        
        response = requests.get(url)
        print(response.status_code)
        print(response.text)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to fetch live data'}), 500
        
        air_data = response.json()
        pollutants = air_data['list'][0]['components']
        
        # Convert to our expected format
        prediction_data = {
            'pm25': pollutants.get('pm2_5', 0),
            'pm10': pollutants.get('pm10', 0),
            'no2': pollutants.get('no2', 0),
            'so2': pollutants.get('so2', 0),
            'co': pollutants.get('co', 0),
            'o3': pollutants.get('o3', 0)
        }
        
        # Make prediction using our model
        prediction_response = predict_aqi_internal(prediction_data)
        # Combine live pollutant data and predicted AQI info
        response_data = prediction_response
        response_data['components'] = pollutants
        return jsonify(response_data)

        
    except Exception as e:
        return jsonify({'error': f'Live AQI fetch failed: {str(e)}'}), 500

def predict_aqi_internal(data):
    """Internal function to make AQI prediction"""
    try:
        pollutants = [
            float(data['pm25']),
            float(data['pm10']),
            float(data['no2']),
            float(data['so2']),
            float(data['co']),
            float(data['o3'])
        ]
        
        features = np.array([pollutants])
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]
        
        aqi_class = label_encoder.inverse_transform([prediction])[0]
        confidence = round(max(model.predict_proba([features])[0]) * 100, 1)
        aqi_value = get_aqi_value_from_class(aqi_class)
        aqi_info = AQI_INFO.get(aqi_class, AQI_INFO['Moderate'])
        
        return {
            'success': True,
            'aqi_class': aqi_class,
            'aqi_value': aqi_value,
            'confidence': round(confidence, 2),
            'emoji': aqi_info['emoji'],
            'color': aqi_info['color'],
            'range': aqi_info['range'],
            'health_tip': aqi_info['health_tip'],
            'pollutants': {
                'pm25': pollutants[0],
                'pm10': pollutants[1],
                'no2': pollutants[2],
                'so2': pollutants[3],
                'co': pollutants[4],
                'o3': pollutants[5]
            },
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {'error': f'Internal prediction failed: {str(e)}'}

if __name__ == '__main__':
    print("🚀 Starting BreatheAware Flask API...")
    app.run(debug=True, host='0.0.0.0', port=5000)