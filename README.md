# Chromosome Karyotype Sorter

An AI-powered system for automatically sorting chromosomes in karyotype images using machine learning.

## Problem Statement

When viewing any species' chromosomes, scientists have to arrange them into pictures called karyotypes. While obtaining the chromosome images isn't difficult, the manual sorting process is extremely tedious and time-consuming.

## Hypothesis

Machine learning algorithms can use **classification** to solve this problem by identifying chromosome types based on morphological features. If classification doesn't work effectively, **regression through measurement** can be used as a fallback approach.

## Features

- 🧬 **Automated Chromosome Detection**: Segments individual chromosomes from karyotype images
- 🎯 **Dual Approach**: Classification-based sorting with regression fallback
- 📊 **Kaggle Integration**: Automatic dataset download from Kaggle
- 🔬 **Advanced Image Processing**: CLAHE enhancement, morphological operations, and feature extraction
- 📈 **Comprehensive Features**: Morphological, texture, and intensity-based features
- 🖼️ **Visualization**: Generate sorted karyotype visualizations
- 🤖 **Multiple ML Models**: Random Forest, with optional PyTorch deep learning support

## Installation

### Quick Setup
```bash
git clone https://github.com/yourusername/chromosome-sorter.git
cd chromosome-sorter
pip install -r requirements.txt
```

### Development Setup
```bash
pip install -e .
```

### Kaggle API Setup (Required for dataset download)
1. Create a Kaggle account at [kaggle.com](https://www.kaggle.com)
2. Go to Account settings and create an API token
3. Download `kaggle.json` and place it in `~/.kaggle/kaggle.json`
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

## Usage

### 1. Download and Setup Data
```bash
python main.py --download --data-dir data
```

### 2. Train Models
```bash
python main.py --train --data-dir data
```

### 3. Sort Chromosomes in Image
```bash
python main.py --image path/to/karyotype.jpg --output sorted_result.png
```

### 4. Use Regression Method
```bash
python main.py --image path/to/karyotype.jpg --method regression
```

## Command Line Options

- `--data-dir`: Directory for storing datasets (default: `data`)
- `--download`: Download chromosome dataset from Kaggle
- `--train`: Train classification and regression models
- `--image`: Path to karyotype image to process
- `--method`: Sorting method (`classification` or `regression`)
- `--output`: Output path for visualization
- `--model-path`: Path to save/load trained models

## How It Works

### 1. Image Preprocessing
- Gaussian blur for noise reduction
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Histogram equalization

### 2. Chromosome Segmentation
- Otsu thresholding
- Morphological operations
- Connected component analysis
- Size and aspect ratio filtering

### 3. Feature Extraction
- **Morphological**: Length, width, area, perimeter, eccentricity
- **Shape**: Aspect ratio, circularity, solidity
- **Texture**: GLCM-based contrast, homogeneity, energy
- **Intensity**: Mean, standard deviation, peak detection

### 4. Classification Approach
- Random Forest classifier with grid search
- Chromosome type prediction (1-22, X, Y)
- Confidence-based sorting

### 5. Regression Fallback
- Predicts chromosome measurements
- Sorts by length and width
- Used when classification fails

## Project Structure

```
chromosome-sorter/
├── main.py                 # Main application
├── Chromosome_Sorter/      # Legacy code (reference)
│   ├── Image_Processor.py
│   ├── Sorter#2.py
│   └── Sorter_#1.py
├── requirements.txt        # Dependencies
├── setup.py               # Installation script
└── README.md              # This file
```

## Dataset

The system uses the **Chromosome Karyotype Images** dataset from Kaggle:
- **Source**: [aliabedimadiseh/chromosome-image-dataset-karyotype](https://www.kaggle.com/datasets/aliabedimadiseh/chromosome-image-dataset-karyotype)
- **Size**: ~1GB
- **Content**: Chromosome images prepared using G-banding technique
- **License**: CC BY-SA 4.0

## Results

The system provides:
- Automated chromosome detection and segmentation
- Classification accuracy metrics
- Sorted chromosome visualizations
- Fallback regression when classification fails

## Technical Details

### Machine Learning Pipeline
1. **Data Loading**: Kaggle API integration
2. **Preprocessing**: Multi-step image enhancement
3. **Segmentation**: Advanced morphological operations
4. **Feature Engineering**: 16+ morphological and texture features
5. **Model Training**: Grid search optimization
6. **Prediction**: Dual-method approach with fallback

### Key Classes
- `ChromosomeDatasetManager`: Handles data download and loading
- `ChromosomeImageProcessor`: Image processing and feature extraction  
- `ChromosomeClassifier`: Classification-based sorting
- `ChromosomeRegressor`: Measurement-based sorting
- `ChromosomeSorter`: Main pipeline orchestration

## Future Improvements

- [ ] Deep learning models (CNN, Vision Transformer)
- [ ] Real-time processing capabilities
- [ ] Multi-species support
- [ ] Web interface
- [ ] Batch processing
- [ ] Advanced visualization tools

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this software in your research, please cite:

```
@software{chromosome_sorter_2025,
  title={Chromosome Karyotype Sorter: AI-powered chromosome sorting system},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/chromosome-sorter}
}
```

## Acknowledgments

- Kaggle community for the chromosome image dataset
- scikit-learn and scikit-image teams
- OpenCV contributors
- All researchers working on computational cytogenetics