from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
import numpy as np
import time

class SVMModel:
    def __init__(self):
        self.model = SVC(probability=True)
        # Espace de recherche réduit et plus ciblé
        self.param_dist = {
            'C': uniform(0.1, 10.0),
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf']  # Focus sur le kernel le plus performant
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
        
        print("Début de l'entraînement SVM...")
        try:
            random_search.fit(X, y)
            
            # Vérification du temps écoulé
            if time.time() - start_time > max_time:
                print("Temps maximum atteint pour SVM")
                
            # Mise à jour du modèle avec les meilleurs paramètres
            self.model = random_search.best_estimator_
            print(f"Meilleurs paramètres SVM : {random_search.best_params_}")
            return random_search.best_score_
            
        except Exception as e:
            print(f"Erreur pendant l'entraînement SVM : {str(e)}")
            # En cas d'erreur, utiliser des paramètres par défaut
            self.model = SVC(probability=True, kernel='rbf', C=1.0, gamma='scale')
            self.model.fit(X, y)
            return self.model.score(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)