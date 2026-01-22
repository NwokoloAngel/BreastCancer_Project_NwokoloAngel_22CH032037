from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load Model
MODEL_RELATIVE_PATH = os.path.join('model', 'breast_cancer_model.pkl')
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_RELATIVE_PATH)

model = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Error: Model not found at {MODEL_PATH}. Please run model/train_model.py first.")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded. Please contact administrator.'}), 500
    
    try:
        # Expected features in specific order:
        # 'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'
        
        feature_keys = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
        features = []
        
        for key in feature_keys:
            val = request.form.get(key)
            if val is None:
                raise ValueError(f"Missing input for {key}")
            features.append(float(val))
            
        # Reshape for prediction (1 sample, 5 features)
        features_arr = np.array(features).reshape(1, -1)
        
        prediction = model.predict(features_arr)[0]
        
        # 0 = Malignant, 1 = Benign (sklearn dataset mapping)
        # Note: We should verify this mapping from training output target_names usually ['malignant', 'benign'] -> 0, 1
        # Wait! sklearn breast cancer `target_names` are ['malignant', 'benign'].
        # BUT usually `target` 0 corresponds to first name?
        # Let's verify standard sklearn behavior:
        # In load_breast_cancer(): 
        # malign = 0, benign = 1.
        
        result_str = "Malignant" if prediction == 0 else "Benign"
        
        return jsonify({
            'prediction': int(prediction), 
            'result': result_str
        })
        
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    # Local dev
    app.run(debug=True, port=5000)
