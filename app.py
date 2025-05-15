# prompt: write a code to give input to content/Data_noiseNshift.h5 as audio

from IPython.display import Audio
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import librosa
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Enable CORS with specific settings
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173", "http://127.0.0.1:5173"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define custom metrics for model loading
def precision(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall

def fscore(y_true, y_pred):
    if tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    f_score = 2 * (p * r) / (p + r + tf.keras.backend.epsilon())
    return f_score

# Load the model at startup
model_path = os.path.join(os.path.dirname(__file__), "Data_noiseNshift.h5")
try:
    custom_objects = {
        'precision': precision,
        'recall': recall,
        'fscore': fscore
    }
    loaded_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    logger.info("Model loaded successfully")
    print('Model input shape:', loaded_model.input_shape)
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    loaded_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_and_resample(audio_path, target_sr=22050, duration=3, offset=0.5):
    """
    Load and resample audio file
    """
    try:
        y, sr = librosa.load(audio_path, sr=target_sr, duration=duration, offset=offset)
        return y, sr
    except Exception as e:
        logger.error(f"Error loading audio: {str(e)}")
        return None, None

def process_single_audio(audio_path, target_sr=22050, duration=3, offset=0.5):
    """
    Process audio file and extract features
    """
    try:
        X, sample_rate = load_and_resample(audio_path, target_sr=target_sr, duration=duration, offset=offset)
        if X is None or sample_rate is None:
            return None

        # Extract MFCC features - using same parameters as training
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=1)
        
        # Ensure we have exactly 259 time steps
        if mfccs.shape[1] < 259:
            pad_width = 259 - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0,0), (0,pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :259]
        
        # Reshape to (259, 1) as expected by the model
        mfccs = mfccs.T  # Now shape is (259, 1)
        return mfccs
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return None

@app.route('/predict', methods=['POST'])
def predict_stress():
    if loaded_model is None:
        logger.error("Model not loaded properly")
        return jsonify({'error': 'Model not loaded properly'}), 500
        
    if 'audio' not in request.files:
        logger.error("No audio file provided")
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process the audio file
            features = process_single_audio(filepath)
            
            if features is not None:
                # Add batch dimension only - shape will be (1, 259, 1)
                features = np.expand_dims(features, axis=0)
                
                # Make prediction
                predictions = loaded_model.predict(features, verbose=0)
                
                # Get the predicted class
                predicted_class = np.argmax(predictions[0])
                
                # Clean up the uploaded file
                os.remove(filepath)
                
                logger.info(f"Successfully processed audio and made prediction: {predicted_class}")
                return jsonify({
                    'predicted_class': int(predicted_class),
                    'prediction_probabilities': predictions[0].tolist()
                })
            else:
                logger.error("Could not process the audio file")
                return jsonify({'error': 'Could not process the audio file'}), 400
                
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
            
    logger.error("Invalid file type")
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
