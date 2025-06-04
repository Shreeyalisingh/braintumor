# Brain Tumor MRI Classification using Deep Learning

A deep learning project for classifying brain tumor types from MRI images using Convolutional Neural Networks (CNN). This project can distinguish between four categories: Glioma, Meningioma, No Tumor, and Pituitary tumors.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project implements a CNN-based classification system for brain tumor detection and classification from MRI images. The model can classify brain MRI scans into four distinct categories:

- **Glioma**: A type of brain tumor that occurs in the brain and spinal cord
- **Meningioma**: A tumor that arises from the meninges
- **No Tumor**: Normal brain MRI scans without tumors
- **Pituitary**: Tumors that develop in the pituitary gland

## 📊 Dataset

The project uses the Brain Tumor MRI Dataset from Kaggle:
- **Source**: `masoudnickparvar/brain-tumor-mri-dataset`
- **Training Images**: 5,712 images across 4 classes
- **Testing Images**: 1,311 images across 4 classes
- **Image Format**: MRI scans in standard image formats
- **Classes**: 4 balanced categories of brain conditions

## ✨ Features

- **Data Visualization**: Distribution analysis and sample image display
- **Image Preprocessing**: Grayscale conversion, resizing, and normalization
- **Object Detection**: Contour-based tumor region detection
- **Data Augmentation**: Rotation, shifting, shearing, zooming, and flipping
- **Deep Learning Model**: Custom CNN architecture with dropout regularization
- **Performance Metrics**: Accuracy, precision, recall, F1-score, and confusion matrix
- **Visual Analysis**: Training curves and prediction visualization

## 🛠️ Requirements

```python
numpy
pandas
seaborn
matplotlib
tensorflow>=2.0
opencv-python
scikit-image
kagglehub
```

## 📦 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Shreeyalisingh/braintumor.git
cd brain-tumor-classification
```

2. **Install required packages:**
```bash
pip install numpy pandas seaborn matplotlib tensorflow opencv-python scikit-image kagglehub
```

3. **Download the dataset:**
The script automatically downloads the dataset using `kagglehub`. Ensure you have a stable internet connection.

## 🚀 Usage

1. **Run the complete pipeline:**
```python
python brain_tumor_classification.py
```

The script will automatically:
- Download and load the dataset
- Perform data exploration and visualization
- Preprocess the images
- Train the CNN model
- Evaluate performance and generate reports

2. **Key Configuration Parameters:**
```python
IMAGE_SIZE = (150, 150)  # Input image dimensions
BATCH_SIZE = 32          # Training batch size
EPOCHS = 10              # Number of training epochs
```

## 🏗️ Model Architecture

The CNN model consists of:

- **Convolutional Layers**: 4 Conv2D layers with ReLU activation
  - Layer 1: 32 filters (3×3)
  - Layer 2: 64 filters (3×3)
  - Layer 3: 128 filters (3×3)
  - Layer 4: 128 filters (3×3)
- **Pooling Layers**: MaxPooling2D (2×2) after each convolution
- **Fully Connected Layers**: 
  - Dense layer with 512 neurons + ReLU
  - Dropout layer (0.5) for regularization
  - Output layer with 4 neurons + Softmax

**Total Parameters**: Varies based on input size
**Optimizer**: Adam
**Loss Function**: Categorical Crossentropy

## 📈 Results

### Model Performance
- **Test Accuracy**: 77.97%
- **Test Loss**: 0.555

### Class-wise Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | 86.09% | 66.00% | 74.72% |
| Meningioma | 63.50% | 54.58% | 58.70% |
| No Tumor | 76.25% | 98.27% | 85.87% |
| Pituitary | 88.85% | 87.67% | 88.26% |

### Key Insights
- The model performs best on "No Tumor" and "Pituitary" classes
- "Meningioma" classification shows room for improvement
- High recall for "No Tumor" indicates good sensitivity for normal cases

```

## 🔧 Customization

### Modify Model Architecture
```python
model = Sequential([
    # Add or modify layers here
    Conv2D(filters, kernel_size, activation='relu'),
    # ... additional layers
])
```

### Adjust Data Augmentation
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,       
    zoom_range=0.2,          
)
```

### Change Training Parameters
```python
epochs = 20              # Train for more epochs
batch_size = 16          # Smaller batch size
image_size = (224, 224)  # Higher resolution
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions or suggestions, please open an issue in the repository.

## 🙏 Acknowledgments

- Dataset provided by Masoud Nickparvar on Kaggle
- TensorFlow and Keras teams for the deep learning framework
- OpenCV community for image processing tools

---

**Note**: This model is for educational and research purposes. For medical applications, please consult with healthcare professionals and ensure proper validation.