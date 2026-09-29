import os
import sys
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# Add src to Python path so we can import preprocessing and feature_extraction
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import preprocess_signals
from src.feature_extraction import extract_features

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'ecg_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'ecg_scaler.pkl')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load model or scaler. Error: {e}")
    model = None
    scaler = None

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler is not loaded on the server."}), 500
        
    try:
        # 1. Parse JSON request
        data = request.json
        if not data:
            return jsonify({"error": "No JSON payload provided."}), 400
            
        threshold = data.get('threshold', 0.50)
        try:
            threshold = float(threshold)
            if threshold <= 0 or threshold >= 1:
                return jsonify({"error": "Threshold must be between 0 and 1 (exclusive)."}), 400
        except ValueError:
            return jsonify({"error": "Invalid threshold value."}), 400
            
        ecg_signal = data.get('ecg_signal')
        if ecg_signal is None:
            return jsonify({"error": "No ECG signal provided in the payload."}), 400
            
        # 2. Parse ECG signal (Expects a list of lists representing [timesteps, leads] or similar)
        # For simplicity in this demo, let's assume it's sent as a 2D array [timesteps, leads]
        ecg_array = np.array(ecg_signal)
        
        # Ensure it has shape (1, num_timesteps, num_leads)
        if len(ecg_array.shape) == 2:
            ecg_array = np.expand_dims(ecg_array, axis=0)
        elif len(ecg_array.shape) != 3:
            return jsonify({"error": "Invalid ECG signal dimensions. Expected 2D or 3D array."}), 400
            
        # 3. Preprocess and Extract Features
        X_preprocessed = preprocess_signals(ecg_array)
        X_features = extract_features(X_preprocessed)
        
        # 4. Scale
        X_scaled = scaler.transform(X_features)
        
        # 5. Predict
        prob = float(model.predict_proba(X_scaled)[0, 1])
        
        # 6. Apply threshold
        prediction_label = "Abnormal ECG pattern" if prob >= threshold else "Normal ECG pattern"
        
        # 7. Return response
        response = {
            "probability": round(prob, 4),
            "threshold": threshold,
            "prediction": prediction_label
        }
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
