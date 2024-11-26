from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import joblib
import os
from utils.preprocessing import preprocess_image

app = Flask(__name__)
CORS(app)

# Chargement des modèles entraînés
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, 'models', 'trained')

# Chargement des modèles et de l'encodeur
svm_model = joblib.load(os.path.join(models_dir, 'svm_model.joblib'))
xgb_model = joblib.load(os.path.join(models_dir, 'xgboost_model.joblib'))
label_encoder = joblib.load(os.path.join(current_dir, 'models', 'label_encoder.joblib'))

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "models": ["svm", "xgboost"]})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupération des données du dessin
        data = request.json
        image_data = np.array(data['image'])
        
        # Prétraitement de l'image
        processed_image = preprocess_image(image_data)
        
        # Mise en forme pour la prédiction
        flattened_image = processed_image.flatten().reshape(1, -1)
        
        # Prédictions des deux modèles
        svm_pred_proba = svm_model.predict_proba(flattened_image)
        xgb_pred_proba = xgb_model.predict_proba(flattened_image)
        
        # Moyenne des probabilités
        avg_proba = (svm_pred_proba + xgb_pred_proba) / 2
        pred_class_idx = np.argmax(avg_proba)
        confidence = float(avg_proba[0][pred_class_idx])
        
        # Conversion de l'index en nom de classe
        predicted_class = label_encoder.inverse_transform([pred_class_idx])[0]
        
        return jsonify({
            "class": predicted_class,
            "confidence": confidence,
            "top_predictions": [
                {
                    "class": label_encoder.inverse_transform([idx])[0],
                    "confidence": float(avg_proba[0][idx])
                }
                for idx in np.argsort(avg_proba[0])[-3:][::-1]  # Top 3 prédictions
            ]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)