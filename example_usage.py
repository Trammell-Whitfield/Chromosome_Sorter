#!/usr/bin/env python3
"""
Example Usage of Chromosome Karyotype Sorter
============================================

This script demonstrates how to use the chromosome sorting system
with different approaches and configurations.
"""

import logging
from pathlib import Path
from main import ChromosomeSorter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_basic_usage():
    """Basic usage example"""
    print("=== Basic Usage Example ===")
    
    # Initialize the sorter
    sorter = ChromosomeSorter(data_dir="data")
    
    # Setup data (download if needed)
    print("Setting up data...")
    sorter.setup_data(download_kaggle=True)
    
    # Train models if they don't exist
    model_path = "chromosome_models.pkl"
    if not Path(model_path).exists():
        print("Training models...")
        sorter.train_models("data")
        sorter.classifier.save_model(model_path)
    else:
        print("Loading existing models...")
        sorter.classifier.load_model(model_path)
    
    # Example image processing (replace with actual image path)
    example_image = "data/example_karyotype.jpg"
    if Path(example_image).exists():
        print(f"Processing {example_image}...")
        
        # Try classification first
        try:
            sorted_chromosomes, predictions = sorter.sort_chromosomes(
                example_image, method='classification'
            )
            print(f"Classification successful! Found {len(sorted_chromosomes)} chromosomes")
            print(f"Predicted types: {predictions}")
            
        except Exception as e:
            print(f"Classification failed: {e}")
            print("Trying regression method...")
            
            # Fallback to regression
            sorted_chromosomes, measurements = sorter.sort_chromosomes(
                example_image, method='regression'
            )
            print(f"Regression successful! Found {len(sorted_chromosomes)} chromosomes")
            print(f"Measurements: {measurements[:5]}...")  # Show first 5
        
        # Create visualization
        sorter.visualize_sorted_chromosomes(
            example_image, 
            output_path="sorted_chromosomes_example.png"
        )
    else:
        print(f"Example image not found at {example_image}")

def example_batch_processing():
    """Example of processing multiple images"""
    print("\n=== Batch Processing Example ===")
    
    sorter = ChromosomeSorter(data_dir="data")
    
    # Load existing models
    model_path = "chromosome_models.pkl"
    if Path(model_path).exists():
        sorter.classifier.load_model(model_path)
        
        # Find all images in data directory
        data_dir = Path("data")
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff'}
        images = [f for f in data_dir.rglob("*") if f.suffix.lower() in image_extensions]
        
        print(f"Found {len(images)} images to process")
        
        results = []
        for i, img_path in enumerate(images[:5]):  # Process first 5 images
            try:
                print(f"Processing {img_path.name}...")
                sorted_chromosomes, predictions = sorter.sort_chromosomes(
                    str(img_path), method='classification'
                )
                
                results.append({
                    'image': img_path.name,
                    'num_chromosomes': len(sorted_chromosomes),
                    'predictions': predictions
                })
                
                # Save individual results
                output_path = f"batch_result_{i}.png"
                sorter.visualize_sorted_chromosomes(str(img_path), output_path)
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
        
        print(f"\nBatch processing complete! Processed {len(results)} images")
        for result in results:
            print(f"- {result['image']}: {result['num_chromosomes']} chromosomes")
    else:
        print("No trained models found. Run training first.")

def example_custom_features():
    """Example of using custom feature extraction"""
    print("\n=== Custom Feature Extraction Example ===")
    
    import cv2
    import numpy as np
    
    # Initialize components
    from main import ChromosomeImageProcessor
    
    processor = ChromosomeImageProcessor()
    
    # Load an example image
    example_image = "data/example_karyotype.jpg"
    if Path(example_image).exists():
        img = cv2.imread(example_image, cv2.IMREAD_GRAYSCALE)
        
        # Preprocess
        processed = processor.preprocess_image(img)
        
        # Segment chromosomes
        chromosomes = processor.segment_chromosomes(processed)
        print(f"Detected {len(chromosomes)} chromosome regions")
        
        # Extract features for each chromosome
        features = processor.extract_morphological_features(chromosomes)
        print(f"Extracted {features.shape[1]} features per chromosome")
        
        # Analyze features
        if len(features) > 0:
            print("\nFeature Statistics:")
            feature_names = [
                'length', 'width', 'area', 'perimeter', 'eccentricity',
                'solidity', 'extent', 'aspect_ratio', 'circularity',
                'num_peaks', 'num_valleys', 'contrast', 'homogeneity',
                'energy', 'mean_intensity', 'std_intensity'
            ]
            
            for i, name in enumerate(feature_names):
                mean_val = np.mean(features[:, i])
                std_val = np.std(features[:, i])
                print(f"  {name}: {mean_val:.3f} ± {std_val:.3f}")
    else:
        print(f"Example image not found at {example_image}")

def example_model_evaluation():
    """Example of evaluating model performance"""
    print("\n=== Model Evaluation Example ===")
    
    from sklearn.model_selection import cross_val_score
    import numpy as np
    
    sorter = ChromosomeSorter(data_dir="data")
    
    # Check if we have data
    data_dir = Path("data")
    if data_dir.exists():
        # Load data
        images, labels, file_paths = sorter.dataset_manager.load_images_from_directory("data")
        
        if len(images) > 10:  # Need sufficient data
            print(f"Evaluating models on {len(images)} images...")
            
            # Extract features from all images
            all_features = []
            all_labels = []
            
            for img, label in zip(images[:50], labels[:50]):  # Use subset for speed
                try:
                    processed_img = sorter.image_processor.preprocess_image(img)
                    chromosome_props = sorter.image_processor.segment_chromosomes(processed_img)
                    
                    if len(chromosome_props) > 0:
                        features = sorter.image_processor.extract_morphological_features(chromosome_props)
                        
                        # Use first chromosome from each image
                        if len(features) > 0:
                            all_features.append(features[0])
                            all_labels.append(label)
                            
                except Exception as e:
                    print(f"Error processing image: {e}")
            
            if len(all_features) > 5:
                all_features = np.array(all_features)
                all_labels = np.array(all_labels)
                
                # Quick cross-validation
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.preprocessing import LabelEncoder, StandardScaler
                
                # Prepare data
                label_encoder = LabelEncoder()
                encoded_labels = label_encoder.fit_transform(all_labels)
                
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(all_features)
                
                # Cross-validation
                rf = RandomForestClassifier(n_estimators=50, random_state=42)
                scores = cross_val_score(rf, scaled_features, encoded_labels, cv=3)
                
                print(f"Cross-validation accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
                print(f"Individual fold scores: {scores}")
            else:
                print("Not enough valid features extracted for evaluation")
        else:
            print("Not enough data for evaluation")
    else:
        print("Data directory not found")

if __name__ == "__main__":
    print("Chromosome Karyotype Sorter - Example Usage")
    print("=" * 50)
    
    # Run examples
    try:
        example_basic_usage()
        example_batch_processing()
        example_custom_features()
        example_model_evaluation()
        
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nExample usage complete!")