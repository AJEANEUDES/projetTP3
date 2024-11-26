# Backend ML pour la Reconnaissance de Dessins

## Configuration

1. Créer un environnement virtuel Python :
```bash
python -m venv venv
source venv/bin/activate  # Sur Unix
venv\Scripts\activate     # Sur Windows
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Structure du Projet

- `app.py` : Point d'entrée de l'application Flask
- `models/` : Implémentations des modèles ML
- `utils/` : Fonctions utilitaires pour le prétraitement

## API Endpoints

- POST `/predict` : Prédit la classe d'un dessin
  - Input : JSON avec une clé 'image' contenant les données du dessin
  - Output : JSON avec la classe prédite et le score de confiance