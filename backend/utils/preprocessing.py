import cv2
import numpy as np
from scipy.ndimage import rotate
from skimage.transform import resize

def preprocess_image(image):
    """
    Prétraite l'image pour la reconnaissance
    """
    # Conversion en niveaux de gris
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Redimensionnement
    image = resize(image, (28, 28), anti_aliasing=True)
    
    # Normalisation
    image = image / 255.0
    
    # Amélioration du contraste
    image = cv2.equalizeHist(np.uint8(image * 255)) / 255.0
    
    return image

def augment_data(image):
    """
    Applique des augmentations de données avancées
    """
    augmented = []
    angles = [-15, -10, -5, 5, 10, 15]
    
    # Rotations
    for angle in angles:
        rotated = rotate(image, angle, reshape=False)
        augmented.append(rotated)
    
    # Ajout de bruit gaussien
    noise = np.random.normal(0, 0.05, image.shape)
    noisy = np.clip(image + noise, 0, 1)
    augmented.append(noisy)
    
    # Variations de contraste
    contrasts = [0.8, 1.2]
    for contrast in contrasts:
        contrasted = np.clip(image * contrast, 0, 1)
        augmented.append(contrasted)
    
    return augmented