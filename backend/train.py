import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils.dataset import DatasetManager
from models.svm_model import SVMModel
from models.xgboost_model import XGBoostModel
import joblib
import os
import time

def main():
    try:
        start_time = time.time()
        print("Début de l'entraînement...")
        
        # Obtention du chemin absolu du dossier backend
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = r"C:\Users\Yao ADJANOHOUN\Documents\Ma maitrise\projet_final\backend\data"
        
        # Vérification et création des dossiers nécessaires
        categories = ['airplane', 'angel', 'ant', 'apple']
        for category in categories:
            category_path = os.path.join(data_dir, category)
            os.makedirs(category_path, exist_ok=True)
            print(f"Dossier créé/vérifié : {category_path}")
        
        # Initialisation du gestionnaire de dataset
        print(f"Utilisation du dossier data : {data_dir}")
        dataset_manager = DatasetManager(data_dir=data_dir)
        
        print("Chargement du dataset...")
        X, y = dataset_manager.load_dataset()
        
        # Encodage des étiquettes
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Sauvegarde de l'encodeur
        joblib.dump(label_encoder, os.path.join(current_dir, 'models', 'label_encoder.joblib'))
        
        # Division en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        
        print(f"Dimensions des données d'entraînement : {X_train.shape}")
        
        # Création du dossier pour sauvegarder les modèles
        models_dir = os.path.join(current_dir, 'models', 'trained')
        os.makedirs(models_dir, exist_ok=True)
        
        # Temps maximum pour chaque modèle (30 minutes)
        max_time_per_model = 1800
        
        # Entraînement et sauvegarde du modèle SVM
        svm_model = SVMModel()
        svm_score = svm_model.train(X_train, y_train, max_time=max_time_per_model)
        print(f"Score SVM: {svm_score:.4f}")
        joblib.dump(svm_model.model, os.path.join(models_dir, 'svm_model.joblib'))
        
        # Entraînement et sauvegarde du modèle XGBoost
        xgb_model = XGBoostModel()
        xgb_score = xgb_model.train(X_train, y_train, max_time=max_time_per_model)
        print(f"Score XGBoost: {xgb_score:.4f}")
        joblib.dump(xgb_model.model, os.path.join(models_dir, 'xgboost_model.joblib'))
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"\nEntraînement terminé en {duration/60:.2f} minutes")
        print("Les modèles ont été sauvegardés.")
        
    except Exception as e:
        print(f"Erreur lors de l'exécution : {str(e)}")
        raise e

if __name__ == "__main__":
    main()