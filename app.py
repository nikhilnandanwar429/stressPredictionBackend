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
CORS(app)  # Enable CORS for all domains

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model at startup
model_path = os.path.join(os.path.dirname(__file__), "stressDetect.h5")
try:
    loaded_model = tf.keras.models.load_model(model_path)
    logger.info("Model loaded successfully")
    print('Model input shape:', loaded_model.input_shape)
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    loaded_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_and_resample(audio_path, target_sr=22050, duration=None, offset=0.3):
    """
    Load and resample audio file
    """
    try:
        y, sr = librosa.load(audio_path, sr=target_sr, duration=duration, offset=offset)
        return y, sr
    except Exception as e:
        logger.error(f"Error loading audio: {str(e)}")
        return None, None

def process_single_audio(audio_path, target_sr=22050, duration=None, offset=0.3):
    """
    Process audio file and extract features
    """
    try:
        X, sample_rate = load_and_resample(audio_path, target_sr=target_sr, duration=duration, offset=offset)
        if X is None or sample_rate is None:
            return None

        # Extract 1 MFCC feature, pad/truncate to 259 frames
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=1)
        desired_frames = 259
        if mfccs.shape[1] < desired_frames:
            pad_width = desired_frames - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0,0),(0,pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :desired_frames]
        # Now mfccs shape is (1, 259)
        mfccs = mfccs.T  # shape (259, 1)
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
                # Reshape the features to match model input shape
                features = np.expand_dims(features, axis=0)  # Add batch dimension
                features = np.expand_dims(features, axis=2)  # Add channel dimension
                
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
    app.run(debug=True)
