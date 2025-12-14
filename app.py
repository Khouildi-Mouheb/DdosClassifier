from flask import Flask, request, jsonify, render_template
import torch
import pickle
import numpy as np
import os

app = Flask(__name__)

# Global variables for loaded models
model = None
transformer = None
label_mapping = None
NUMBER_OF_FEATURES = None

def load_models():
    """Load trained models and metadata."""
    global model, transformer, label_mapping, NUMBER_OF_FEATURES
    
    try:
        # Load the transformer first to get feature count
        with open("transformer.pkl", "rb") as f:
            transformer = pickle.load(f)
        
        # Load label mapping
        with open("label_mapping.pkl", "rb") as f:
            label_mapping = pickle.load(f)
        
        # Inverse mapping for display
        inverse_mapping = {v: k for k, v in label_mapping.items()}
        
        # Load the model
        with open("classificationModel.pkl", "rb") as f:
            model = pickle.load(f)
        
        # Get number of features from model or transformer
        if hasattr(model, 'layerstack'):
            # Get from first layer of model
            first_layer = model.layerstack[0]
            NUMBER_OF_FEATURES = first_layer.in_features
        else:
            # Default or estimate
            NUMBER_OF_FEATURES = 78  # Update with your actual count
        
        print(f"✓ Models loaded successfully")
        print(f"  - Number of features: {NUMBER_OF_FEATURES}")
        print(f"  - Label mapping: {label_mapping}")
        
        return True
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return False

# Load models on startup
if load_models():
    print("Application ready!")
else:
    print("Warning: Models not loaded. Run train.py first.")

@app.route('/')
def index():
    """Serve the main educational page."""
    return render_template('index.html', 
                          num_features=NUMBER_OF_FEATURES if NUMBER_OF_FEATURES else 78,
                          label_mapping=label_mapping)

@app.route('/predict', methods=["POST"])
def predict():
    """API endpoint for full-feature predictions."""
    if model is None:
        return jsonify({"error": "Model not loaded", "status": "failed"}), 500
    
    try:
        data = request.get_json()
        
        if "features" not in data:
            return jsonify({"error": "No features provided", "status": "failed"}), 400
        
        features = data["features"]
        
        # Validate input length
        if len(features) != NUMBER_OF_FEATURES:
            return jsonify({
                "error": f"Invalid input. Expected {NUMBER_OF_FEATURES} features, got {len(features)}.",
                "expected": NUMBER_OF_FEATURES,
                "received": len(features),
                "status": "failed"
            }), 400
        
        # Convert to tensor
        features_np = np.array(features, dtype=np.float32).reshape(1, -1)
        features_tensor = torch.tensor(features_np, dtype=torch.float32)
        
        # Normalize if transformer is available
        if transformer:
            features_tensor = transformer.normalize_new_data(features_tensor)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            logits = model(features_tensor)
            probability = torch.sigmoid(logits).item()
            is_attack = probability > 0.5
        
        # Get label name
        prediction_label = "ATTACK" if is_attack else "NORMAL"
        label_value = 1 if is_attack else 0
        
        if label_mapping and label_value in label_mapping.values():
            # Find the key for this value
            for key, val in label_mapping.items():
                if val == label_value:
                    prediction_label = key
                    break
        
        return jsonify({
            "prediction": prediction_label,
            "probability": probability,
            "confidence": f"{(probability * 100):.2f}%" if is_attack else f"{((1 - probability) * 100):.2f}%",
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500

@app.route('/test_simple', methods=["POST"])
def test_simple():
    """Simplified endpoint for web interface with partial features."""
    if model is None:
        return jsonify({"error": "Model not loaded", "status": "failed"}), 500
    
    try:
        data = request.get_json()
        user_features = data.get("features", [])
        
        # Pad with zeros if user provided fewer features
        if len(user_features) < NUMBER_OF_FEATURES:
            user_features = user_features + [0.0] * (NUMBER_OF_FEATURES - len(user_features))
        elif len(user_features) > NUMBER_OF_FEATURES:
            user_features = user_features[:NUMBER_OF_FEATURES]
        
        # Convert to tensor
        features_np = np.array(user_features, dtype=np.float32).reshape(1, -1)
        features_tensor = torch.tensor(features_np, dtype=torch.float32)
        
        # Normalize if transformer is available
        if transformer:
            features_tensor = transformer.normalize_new_data(features_tensor)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            logits = model(features_tensor)
            probability = torch.sigmoid(logits).item()
            is_attack = probability > 0.5
        
        # Generate explanation
        if is_attack:
            explanation = (
                "🚨 DDoS ATTACK DETECTED! The model identified patterns consistent with a Distributed Denial of Service attack. "
                "This could include unusually high packet rates, short flow durations, or abnormal protocol behavior that overwhelms "
                "network resources. Immediate investigation is recommended."
            )
        else:
            explanation = (
                "✅ NORMAL TRAFFIC. The analyzed patterns appear consistent with legitimate network activity. "
                "Flow characteristics match expected user behavior without signs of malicious flooding or resource exhaustion attempts."
            )
        
        # Add confidence level
        confidence = probability if is_attack else (1 - probability)
        confidence_level = "HIGH" if confidence > 0.8 else "MEDIUM" if confidence > 0.6 else "LOW"
        
        return jsonify({
            "prediction": "ATTACK" if is_attack else "NORMAL",
            "probability": probability,
            "confidence": confidence_level,
            "explanation": explanation,
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500

@app.route('/model_info', methods=["GET"])
def model_info():
    """Get information about the loaded model."""
    if model is None:
        return jsonify({"error": "Model not loaded", "status": "failed"}), 500
    
    info = {
        "num_features": NUMBER_OF_FEATURES,
        "label_mapping": label_mapping,
        "model_architecture": str(model),
        "status": "success"
    }
    
    return jsonify(info)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)