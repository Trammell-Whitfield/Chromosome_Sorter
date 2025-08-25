import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology, feature
from scipy import signal, stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load images
def load_karyotype_images(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img = cv2.imread(os.path.join(directory, filename))
            if img is not None:
                images.append(img)
                # Assuming the filename contains the label/classification
                labels.append(filename.split('_')[0])
    return np.array(images), np.array(labels)

class ChromosomeClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def preprocess_image(self, image_path):
        logging.info(f"Preprocessing image: {image_path}")
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Unable to read image: {image_path}")
        
        # Apply various preprocessing techniques
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.equalizeHist(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        
        return img

    def segment_chromosomes(self, img):
        logging.info("Segmenting chromosomes")
        thresh = filters.threshold_otsu(img)
        binary = img > thresh
        binary = morphology.remove_small_objects(binary, min_size=100)
        binary = morphology.closing(binary, morphology.disk(3))
        labeled = measure.label(binary)
        props = measure.regionprops(labeled, img)
        return props

    def extract_features(self, chromosome):
        # Basic geometric features
        length = chromosome.major_axis_length
        width = chromosome.minor_axis_length
        area = chromosome.area
        perimeter = chromosome.perimeter
        eccentricity = chromosome.eccentricity
        
        # Intensity features
        intensity_profile = chromosome.intensity_image.mean(axis=1)
        mean_intensity = np.mean(intensity_profile)
        std_intensity = np.std(intensity_profile)
        
        # Texture features
        glcm = feature.graycomatrix(chromosome.intensity_image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
        contrast = feature.graycoprops(glcm, 'contrast').mean()
        dissimilarity = feature.graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = feature.graycoprops(glcm, 'homogeneity').mean()
        energy = feature.graycoprops(glcm, 'energy').mean()
        correlation = feature.graycoprops(glcm, 'correlation').mean()
        
        # Cytogenetic mapping
        peaks, _ = signal.find_peaks(intensity_profile, distance=10)
        valleys, _ = signal.find_peaks(-intensity_profile, distance=10)
        
        # Combine all features
        features = np.array([length, width, area, perimeter, eccentricity,
                             mean_intensity, std_intensity, contrast, dissimilarity,
                             homogeneity, energy, correlation,
                             len(peaks), len(valleys)])
        
        return features

    def train_model(self, features, labels):
        logging.info("Training model")
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize RandomForestClassifier
        rf = RandomForestClassifier(random_state=42)
        
        # Define parameter grid for GridSearchCV
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]}
        
        # Create GridSearchCV
        grid_search = GridSearchCV(rf, param_grid, cv=5)
        
        # Fit grid search
        grid_search.fit(X_train_scaled, y_train)
        
        # Set the best model
        self.model = grid_search.best_estimator_
        
        # Evaluate the best model on the test set
        y_pred = self.model.predict(X_test_scaled)
        logging.info("Model training complete")
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        
        return self.model

# Load images from a directory (replace with actual path)
images, labels = load_karyotype_images('path/to/your/karyotype_images/')
print(f"Loaded {len(images)} karyotype images")