import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import numpy as np
import time

class XGBoostModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            objective='multi:softprob',
            tree_method='hist',  # Méthode plus rapide
            enable_categorical=True
        )
        # Espace de recherche réduit
        self.param_dist = {
            'max_depth': randint(3, 6),
            'learning_rate': uniform(0.01, 0.2),
            'n_estimators': randint(50, 150),
            'min_child_weight': randint(1, 4)
        }
        
    def train(self, X, y, max_time=3600):  # 1 heure par défaut
        start_time = time.time()
        
        # Configuration de la recherche aléatoire
        random_search = RandomizedSearchCV(
            self.model,
            self.param_dist,
            n_iter=10,  # Nombre limité d'itérations
            cv=3,
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        print("Début de l'entraînement XGBoost...")
        try:
            random_search.fit(X, y)
            
            # Vérification du temps écoulé
            if time.time() - start_time > max_time:
                print("Temps maximum atteint pour XGBoost")
                
            # Mise à jour du modèle avec les meilleurs paramètres
            self.model = random_search.best_estimator_
            print(f"Meilleurs paramètres XGBoost : {random_search.best_params_}")
            return random_search.best_score_
            
        except Exception as e:
            print(f"Erreur pendant l'entraînement XGBoost : {str(e)}")
            # En cas d'erreur, utiliser des paramètres par défaut
            self.model = xgb.XGBClassifier(
                objective='multi:softprob',
                tree_method='hist',
                max_depth=3,
                learning_rate=0.1,
                n_estimators=100
            )
            self.model.fit(X, y)
            return self.model.score(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)