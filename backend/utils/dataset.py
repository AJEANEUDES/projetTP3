import os
import numpy as np
from PIL import Image

class DatasetManager:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        
    def load_dataset(self):
        """
        Charge le dataset à partir du répertoire local
        """
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Le répertoire {self.data_dir} n'existe pas.")
            
        X = []
        y = []
        
        # Parcours des catégories (dossiers)
        categories = [d for d in os.listdir(self.data_dir) 
                     if os.path.isdir(os.path.join(self.data_dir, d))]
        
        if not categories:
            raise ValueError(f"Aucune catégorie trouvée dans {self.data_dir}")
            
        print(f"Catégories trouvées : {categories}")
        
        # Parcours des catégories (dossiers)
        for category in categories:
            category_path = os.path.join(self.data_dir, category)
            print(f"Chargement de la catégorie: {category} depuis {category_path}")
            
            # Parcours des images dans chaque catégorie
            images = [f for f in os.listdir(category_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not images:
                print(f"Attention: Aucune image trouvée dans {category_path}")
                continue
                
            for filename in images:
                try:
                    img_path = os.path.join(category_path, filename)
                    print(f"Chargement de l'image: {img_path}")
                    img = Image.open(img_path)
                    img_array = self._preprocess_image(img)
                    X.append(img_array)
                    y.append(category)
                except Exception as e:
                    print(f"Erreur lors du chargement de {filename}: {str(e)}")
                    continue
        
        if not X:
            raise ValueError("Aucune image n'a pu être chargée.")
        
        print(f"Dataset chargé avec succès. {len(X)} images trouvées.")
        return np.array(X), np.array(y)
    
    def _preprocess_image(self, img):
        """
        Prétraite une image pour l'apprentissage
        """
        # Conversion en niveaux de gris
        img = img.convert('L')
        # Redimensionnement
        img = img.resize((28, 28))
        # Normalisation
        img_array = np.array(img) / 255.0
        return img_array.flatten()