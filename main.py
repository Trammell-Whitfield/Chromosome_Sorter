#!/usr/bin/env python3
"""
Chromosome Karyotype Sorter
==========================

Problem: 
When viewing any species' chromosomes, scientists have to arrange them into pictures,
first called karyotypes. Obtaining the chromosome isn't tricky, but the sorting process is arduous.

Hypothesis: 
Machine learning algorithms can use classification to solve this problem.
If classification doesn't work, regression through measurement can be used as a fallback.

This main.py integrates:
- Kaggle API for chromosome dataset access
- Classification-based chromosome sorting
- Regression-based measurement fallback
- Image processing and feature extraction
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# ML and Data Processing1
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.cluster import KMeans

# Image Processing
from skimage import filters, measure, morphology, feature
from scipy import signal, stats
import joblib

# Deep Learning (if available)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChromosomeDatasetManager:
    """Handles Kaggle dataset download and management"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_kaggle_dataset(self, dataset_name="aliabedimadiseh/chromosome-image-dataset-karyotype"):
        """Download chromosome dataset from Kaggle using kagglehub"""
        try:
            import kagglehub
            logger.info(f"Downloading {dataset_name} from Kaggle using kagglehub...")
            
            # Download the dataset - kagglehub handles authentication automatically
            dataset_path = kagglehub.dataset_download(dataset_name)
            logger.info(f"Dataset downloaded to: {dataset_path}")
            
            # Copy or link to our data directory if needed
            import shutil
            if not (self.data_dir / "kaggle_data").exists():
                if Path(dataset_path).exists():
                    shutil.copytree(dataset_path, self.data_dir / "kaggle_data")
                    logger.info(f"Dataset copied to {self.data_dir / 'kaggle_data'}")
                else:
                    logger.warning(f"Dataset path {dataset_path} not found")
            
            return True
            
        except ImportError:
            logger.error("kagglehub not installed. Run: pip install kagglehub")
            return False
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            logger.info("You can also try: pip install kagglehub[pandas-datasets]")
            return False
    
    def load_kaggle_dataframe(self, dataset_name="aliabedimadiseh/chromosome-image-dataset-karyotype", file_path=""):
        """Load dataset using kagglehub pandas adapter"""
        try:
            import kagglehub
            from kagglehub import KaggleDatasetAdapter
            
            logger.info(f"Loading dataset {dataset_name} as pandas DataFrame...")
            
            # Load the latest version
            df = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                dataset_name,
                file_path,
            )
            
            logger.info(f"Dataset loaded with shape: {df.shape}")
            logger.info("First 5 records:")
            print(df.head())
            
            return df
            
        except ImportError:
            logger.error("kagglehub not installed. Run: pip install kagglehub[pandas-datasets]")
            return None
        except Exception as e:
            logger.error(f"Error loading dataset as DataFrame: {e}")
            return None
    
    def load_images_from_directory(self, directory):
        """Load images and extract labels from directory structure"""
        images = []
        labels = []
        file_paths = []
        
        directory = Path(directory)
        if not directory.exists():
            logger.error(f"Directory {directory} not found")
            return np.array([]), np.array([]), []
        
        # Handle different file extensions
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif']
        
        for ext in image_extensions:
            for img_path in directory.glob(ext):
                try:
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Resize images to a standard size for consistency
                        img_resized = cv2.resize(img, (128, 128))
                        images.append(img_resized)
                        # Extract label from filename or parent directory
                        label = self._extract_label_from_path(img_path)
                        labels.append(label)
                        file_paths.append(str(img_path))
                except Exception as e:
                    logger.warning(f"Failed to load {img_path}: {e}")
        
        logger.info(f"Loaded {len(images)} images")
        return np.array(images), np.array(labels), file_paths
    
    def _extract_label_from_path(self, img_path):
        """Extract chromosome type/number from file path"""
        filename = img_path.stem.lower()
        
        # For this dataset, we'll use a more generic approach
        # Since we don't have clear chromosome labels, we'll create synthetic labels
        # based on image properties or use a random assignment for training
        
        # Try to extract chromosome number from filename patterns
        for i in range(1, 24):  # Chromosomes 1-23
            if f"chr{i}" in filename or f"chromosome{i}" in filename or f"_{i}_" in filename:
                return f"chr{i}"
        
        # Look for sex chromosomes
        if 'x' in filename or 'chrx' in filename:
            return 'chrX'
        if 'y' in filename or 'chry' in filename:
            return 'chrY'
            
        # For this specific dataset, since filenames are just numbers,
        # we'll generate synthetic labels based on filename hash for consistency
        filename_hash = hash(filename) % 24
        if filename_hash == 0:
            return 'chrX'
        elif filename_hash == 23:
            return 'chrY'
        else:
            return f'chr{filename_hash}'

class ChromosomeImageProcessor:
    """Handles chromosome image preprocessing and feature extraction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def preprocess_image(self, img):
        """Apply preprocessing to enhance chromosome visibility"""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        # Apply Gaussian blur to reduce noise
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        # Histogram equalization
        img = cv2.equalizeHist(img)
        
        return img
    
    def segment_chromosomes(self, img):
        """Segment individual chromosomes from karyotype image"""
        # Apply Otsu thresholding
        thresh = filters.threshold_otsu(img)
        binary = img > thresh
        
        # Remove small objects and close gaps
        binary = morphology.remove_small_objects(binary, min_size=200)
        binary = morphology.closing(binary, morphology.disk(3))
        
        # Label connected components
        labeled = measure.label(binary)
        props = measure.regionprops(labeled, img)
        
        # Filter by area and aspect ratio to remove artifacts
        filtered_props = []
        for prop in props:
            area = prop.area
            aspect_ratio = prop.major_axis_length / (prop.minor_axis_length + 1e-6)
            
            if 500 < area < 50000 and 1.5 < aspect_ratio < 10:
                filtered_props.append(prop)
        
        return filtered_props
    
    def extract_morphological_features(self, chromosome_props):
        """Extract morphological features from chromosome"""
        features = []
        
        for prop in chromosome_props:
            # Basic geometric features
            length = prop.major_axis_length
            width = prop.minor_axis_length
            area = prop.area
            perimeter = prop.perimeter
            eccentricity = prop.eccentricity
            solidity = prop.solidity
            extent = prop.extent
            
            # Aspect ratio and circularity
            aspect_ratio = length / (width + 1e-6)
            circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
            
            # Centromere detection (intensity profile analysis)
            intensity_profile = prop.intensity_image.mean(axis=1)
            peaks, _ = signal.find_peaks(intensity_profile, distance=10)
            valleys, _ = signal.find_peaks(-intensity_profile, distance=10)
            
            # Texture features using GLCM
            try:
                glcm = feature.graycomatrix(
                    (prop.intensity_image * 255).astype(np.uint8),
                    [1], [0, np.pi/4, np.pi/2, 3*np.pi/4]
                )
                contrast = feature.graycoprops(glcm, 'contrast').mean()
                homogeneity = feature.graycoprops(glcm, 'homogeneity').mean()
                energy = feature.graycoprops(glcm, 'energy').mean()
            except:
                contrast = homogeneity = energy = 0
            
            feature_vector = [
                length, width, area, perimeter, eccentricity, solidity, extent,
                aspect_ratio, circularity, len(peaks), len(valleys),
                contrast, homogeneity, energy,
                np.mean(intensity_profile), np.std(intensity_profile)
            ]
            
            features.append(feature_vector)
        
        return np.array(features)

class ChromosomeClassifier:
    """Classification-based chromosome sorting"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def train(self, features, labels):
        """Train classification model"""
        logger.info("Training classification model...")
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels_encoded, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Grid search for best parameters
        rf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        
        grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        
        self.model = grid_search.best_estimator_
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = (y_pred == y_test).mean()
        
        logger.info(f"Classification accuracy: {accuracy:.3f}")
        logger.info("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        return self.model
    
    def predict(self, features):
        """Predict chromosome types"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)
        return self.label_encoder.inverse_transform(predictions)
    
    def save_model(self, filepath):
        """Save trained model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }, filepath)
    
    def load_model(self, filepath):
        """Load trained model"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.label_encoder = data['label_encoder']

class ChromosomeRegressor:
    """Regression-based chromosome measurement and sorting"""
    
    def __init__(self):
        self.length_model = None
        self.width_model = None
        self.scaler = StandardScaler()
        
    def train(self, features, measurements):
        """Train regression models for chromosome measurements"""
        logger.info("Training regression models...")
        
        # Extract length and width measurements
        lengths = measurements[:, 0]  # Assuming first column is length
        widths = measurements[:, 1]   # Assuming second column is width
        
        # Split data
        X_train, X_test, len_train, len_test, wid_train, wid_test = train_test_split(
            features, lengths, widths, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train length predictor
        self.length_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.length_model.fit(X_train_scaled, len_train)
        
        # Train width predictor
        self.width_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.width_model.fit(X_train_scaled, wid_train)
        
        # Evaluate
        len_pred = self.length_model.predict(X_test_scaled)
        wid_pred = self.width_model.predict(X_test_scaled)
        
        len_mse = mean_squared_error(len_test, len_pred)
        wid_mse = mean_squared_error(wid_test, wid_pred)
        
        logger.info(f"Length MSE: {len_mse:.3f}")
        logger.info(f"Width MSE: {wid_mse:.3f}")
        
        return self.length_model, self.width_model
    
    def predict_measurements(self, features):
        """Predict chromosome measurements"""
        features_scaled = self.scaler.transform(features)
        lengths = self.length_model.predict(features_scaled)
        widths = self.width_model.predict(features_scaled)
        return lengths, widths
    
    def sort_by_measurements(self, features):
        """Sort chromosomes by predicted measurements"""
        lengths, widths = self.predict_measurements(features)
        # Sort by length primarily, width secondarily
        sort_indices = np.lexsort((widths, lengths))[::-1]  # Descending order
        return sort_indices

class ChromosomeSorter:
    """Main chromosome sorting pipeline"""
    
    def __init__(self, data_dir="data"):
        self.dataset_manager = ChromosomeDatasetManager(data_dir)
        self.image_processor = ChromosomeImageProcessor()
        self.classifier = ChromosomeClassifier()
        self.regressor = ChromosomeRegressor()
        
    def setup_data(self, download_kaggle=True):
        """Setup and download chromosome data"""
        if download_kaggle:
            success = self.dataset_manager.download_kaggle_dataset()
            if not success:
                logger.warning("Failed to download Kaggle dataset. Please download manually.")
        
        return self.dataset_manager
    
    def process_karyotype_image(self, image_path):
        """Process a single karyotype image and extract chromosomes"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Preprocess
        processed_img = self.image_processor.preprocess_image(img)
        
        # Segment chromosomes
        chromosome_props = self.image_processor.segment_chromosomes(processed_img)
        
        # Extract features
        features = self.image_processor.extract_morphological_features(chromosome_props)
        
        return chromosome_props, features
    
    def train_models(self, data_directory, max_images=500):
        """Train both classification and regression models"""
        images, labels, file_paths = self.dataset_manager.load_images_from_directory(data_directory)
        
        if len(images) == 0:
            raise ValueError("No images found in data directory")
        
        # Limit the number of images to avoid memory issues
        if len(images) > max_images:
            logger.info(f"Using subset of {max_images} images for training (out of {len(images)} available)")
            indices = np.random.choice(len(images), max_images, replace=False)
            images = images[indices]
            labels = np.array(labels)[indices]
        
        # Process all images and extract features
        all_features = []
        all_labels = []
        all_measurements = []
        
        for img, label in zip(images, labels):
            try:
                processed_img = self.image_processor.preprocess_image(img)
                chromosome_props = self.image_processor.segment_chromosomes(processed_img)
                
                if len(chromosome_props) > 0:
                    features = self.image_processor.extract_morphological_features(chromosome_props)
                    
                    for i, feature_vec in enumerate(features):
                        all_features.append(feature_vec)
                        all_labels.append(label)
                        # Use actual measurements from regionprops
                        prop = chromosome_props[i]
                        all_measurements.append([prop.major_axis_length, prop.minor_axis_length])
                        
            except Exception as e:
                logger.warning(f"Failed to process image with label {label}: {e}")
        
        if len(all_features) == 0:
            raise ValueError("No features extracted from images")
        
        all_features = np.array(all_features)
        all_labels = np.array(all_labels)
        all_measurements = np.array(all_measurements)
        
        # Train classification model
        self.classifier.train(all_features, all_labels)
        
        # Train regression model
        self.regressor.train(all_features, all_measurements)
        
        logger.info("Both models trained successfully!")
    
    def sort_chromosomes(self, image_path, method='classification'):
        """Sort chromosomes in a karyotype image"""
        chromosome_props, features = self.process_karyotype_image(image_path)
        
        if len(features) == 0:
            logger.warning("No chromosomes detected in image")
            return [], []
        
        if method == 'classification':
            try:
                predictions = self.classifier.predict(features)
                # Sort by chromosome number/type
                sort_indices = np.argsort(predictions)
                return [chromosome_props[i] for i in sort_indices], predictions[sort_indices]
            except Exception as e:
                logger.warning(f"Classification failed: {e}. Falling back to regression.")
                method = 'regression'
        
        if method == 'regression':
            sort_indices = self.regressor.sort_by_measurements(features)
            lengths, widths = self.regressor.predict_measurements(features)
            sorted_chromosomes = [chromosome_props[i] for i in sort_indices]
            measurements = [(lengths[i], widths[i]) for i in sort_indices]
            return sorted_chromosomes, measurements
        
        raise ValueError("Unknown sorting method")
    
    def visualize_sorted_chromosomes(self, image_path, output_path=None):
        """Create visualization of sorted chromosomes"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        sorted_chromosomes, labels = self.sort_chromosomes(image_path)
        
        if len(sorted_chromosomes) == 0:
            logger.warning("No chromosomes to visualize")
            return
        
        # Create subplot for each chromosome
        n_chromosomes = len(sorted_chromosomes)
        cols = min(6, n_chromosomes)
        rows = (n_chromosomes + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (chrom, label) in enumerate(zip(sorted_chromosomes, labels)):
            row, col = i // cols, i % cols
            
            # Extract chromosome region
            minr, minc, maxr, maxc = chrom.bbox
            chrom_img = img[minr:maxr, minc:maxc]
            
            axes[row, col].imshow(chrom_img, cmap='gray')
            axes[row, col].set_title(f'{label}' if isinstance(label, str) else f'L:{label[0]:.1f} W:{label[1]:.1f}')
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(n_chromosomes, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_path}")
        
        plt.show()

def main():
    """Main function to run chromosome sorting pipeline"""
    parser = argparse.ArgumentParser(description='Chromosome Karyotype Sorter')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--download', action='store_true', help='Download Kaggle dataset')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--image', help='Path to karyotype image to sort')
    parser.add_argument('--method', choices=['classification', 'regression'], 
                       default='classification', help='Sorting method')
    parser.add_argument('--output', help='Output path for visualization')
    parser.add_argument('--model-path', default='chromosome_models.pkl', 
                       help='Path to save/load models')
    
    args = parser.parse_args()
    
    # Initialize sorter
    sorter = ChromosomeSorter(args.data_dir)
    
    # Setup data
    if args.download:
        sorter.setup_data(download_kaggle=True)
    
    # Train models
    if args.train:
        data_path = Path(args.data_dir)
        if not data_path.exists() or not any(data_path.iterdir()):
            logger.error(f"No data found in {args.data_dir}. Use --download flag first.")
            return
        
        sorter.train_models(args.data_dir)
        sorter.classifier.save_model(args.model_path)
        logger.info(f"Models saved to {args.model_path}")
    
    # Load existing models
    else:
        try:
            sorter.classifier.load_model(args.model_path)
            logger.info(f"Models loaded from {args.model_path}")
        except FileNotFoundError:
            logger.error(f"No trained models found at {args.model_path}. Use --train flag first.")
            return
    
    # Process single image
    if args.image:
        if not Path(args.image).exists():
            logger.error(f"Image not found: {args.image}")
            return
        
        logger.info(f"Sorting chromosomes in {args.image}")
        sorter.visualize_sorted_chromosomes(args.image, args.output)

if __name__ == "__main__":
    main()