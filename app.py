from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
from dotenv import load_dotenv
import warnings

# Suppress scikit-learn version mismatch warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Load environment variables
load_dotenv()

# Get port from environment variable, default to 5000 if not set
PORT = int(os.getenv('PORT', 5000))

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model
try:
    with open('iris.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        required_fields = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
            
        arr = np.array([
            [float(data['sepal_length']), float(data['sepal_width']), 
            float(data['petal_length']), float(data['petal_width'])]
        ])
        pred = model.predict(arr)
        probabilities = model.predict_proba(arr)[0]
        
        species_names = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
        species_probabilities = {
            species_names[i]: float(probabilities[i])
            for i in range(len(species_names))
        }
        
        return jsonify({
            'species': species_names[int(pred[0])],
            'confidence': float(max(probabilities)),
            'probabilities': species_probabilities
        })
    except ValueError as e:
        return jsonify({'error': 'Invalid input data'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=PORT, debug=debug)