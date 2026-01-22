# 🗑️ Waste Classification using Transfer Learning with VGG16

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17.0-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Latest-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Final Project Course:** Deep Learning with Keras and TensorFlow (Coursera)  
> **Technology:** Transfer Learning with pre-trained VGG16 model  
> **Application:** Automatic waste classification (Organic vs. Recyclable)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Objectives](#project-objectives)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [How to Use](#how-to-use)
- [Results](#results)
- [Methodology](#methodology)
- [Contributions](#contributions)
- [License](#license)
- [References](#references)

---

## 🎯 Overview

This project implements an automatic waste classification system using advanced **Deep Learning** and **Transfer Learning** techniques. The model can distinguish between **organic** (O) and **recyclable** (R) waste through image analysis, using the VGG16 architecture pre-trained on the ImageNet dataset.

### Problem Context

Manual waste classification is a laborious, error-prone process that can lead to contamination of recyclable materials. This project aims to automate this process using computer vision and machine learning, improving efficiency and reducing contamination rates in waste management systems.

### Proposed Solution

Using **Transfer Learning** with the pre-trained VGG16 model, we developed two models:

1. **Extract Features Model**: Model that extracts features using frozen VGG16 layers
2. **Fine-Tuned Model**: Refined model with fine-tuning of the last VGG16 layers

---

## 🎯 Project Objectives

Upon completion of this project, you will be able to:

- ✅ Apply **Transfer Learning** using the VGG16 model for image classification
- ✅ Prepare and preprocess image data for machine learning tasks
- ✅ Perform **fine-tuning** of a pre-trained model to improve accuracy
- ✅ Evaluate model performance using appropriate metrics
- ✅ Visualize model predictions on test data

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Transfer Learning** | Leveraging pre-trained knowledge from VGG16 |
| **Data Augmentation** | Data augmentation for better generalization |
| **Two Models** | Comparison between extract features and fine-tuning |
| **GPU Optimization** | Configured for acceleration on Intel Arc GPU |
| **Visualizations** | Loss and accuracy plots during training |
| **Model Checkpointing** | Automatic saving of best models |
| **Early Stopping** | Overfitting prevention during training |
| **Learning Rate Decay** | Exponential decay of learning rate |

---

## 📊 Dataset

### Source

The dataset used is the [Waste Classification Dataset](https://www.kaggle.com/datasets/techsash/waste-classification-data) available on Kaggle.

### Structure

```
o-vs-r-split/
├── train/
│   ├── O/          # 500 organic waste images
│   └── R/          # 500 recyclable waste images
└── test/
    ├── O/          # 100 organic waste images
    └── R/          # 100 recyclable waste images
```

### Statistics

| Metric | Value |
|--------|-------|
| **Total images** | 1,200 |
| **Training** | 1,000 images (800 train + 200 validation) |
| **Test** | 200 images |
| **Classes** | 2 (Organic 'O' and Recyclable 'R') |
| **Dimensions** | 150x150 pixels |
| **Format** | JPG |

### Data Split

- **Train**: 80% (800 images)
- **Validation**: 20% of training set (200 images)
- **Test**: 200 images (100 per class)

---

## 🏗️ Model Architecture

### Base Model: VGG16

The model uses the VGG16 architecture pre-trained on ImageNet as a base:

```
VGG16 Base (Frozen)
├── Convolutional Blocks (frozen)
│   ├── Conv2D + ReLU
│   ├── MaxPooling2D
│   └── ... (13 convolutional layers)
└── Dense Layers (trainable)
    ├── Flatten
    ├── Dense(512) + ReLU + Dropout(0.5)
    ├── Dense(512) + ReLU + Dropout(0.5)
    └── Dense(1) + Sigmoid (binary output)
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| **Total parameters** | 19,172,673 |
| **Trainable parameters** | 4,457,985 (Extract Features Model) |
| **Non-trainable parameters** | 14,714,688 (frozen VGG16 layers) |
| **Model size** | ~73.14 MB |

### Fine-Tuning

In the fine-tuned model, the last convolutional layers of VGG16 are unfrozen and trained with a lower learning rate.

---

## 🛠️ Technologies Used

### Main Libraries

- **TensorFlow** 2.17.0 - Deep learning framework
- **Keras** - High-level API for TensorFlow
- **NumPy** 1.26.4 - Mathematical operations and arrays
- **scikit-learn** 1.5.1 - Evaluation metrics
- **Matplotlib** 3.9.2 - Data visualization and plots

### Extensions and Optimizations

- **Intel Extension for TensorFlow** - Acceleration on Intel Arc GPU
- **ImageDataGenerator** - Real-time data generation and augmentation

### Optimized Hardware

- Intel Arc GPU (optional, but recommended for better performance)

---

## 💻 Installation

### Prerequisites

- Python 3.12 or higher
- pip (Python package manager)
- Git (to clone the repository)

### Step by Step

**1. Clone the repository**

```bash
git clone https://github.com/sidnei-almeida/waste-classification-vgg16.git
cd waste-classification-vgg16
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install tensorflow==2.17.0
pip install numpy==1.26.4
pip install scikit-learn==1.5.1
pip install matplotlib==3.9.2
pip install intel-extension-for-tensorflow[xpu]  # Optional, for Intel Arc GPU
```

**4. Download the dataset**

The dataset will be downloaded automatically when running the notebook. Alternatively, you can download it manually using the code provided in the notebook.

---

## 📁 Project Structure

```
waste-classification-vgg16/
│
├── README.md                                    # This file
├── Final Proj-Classify Waste Products Using TL- FT-v1.ipynb  # Main notebook
│
├── o-vs-r-split/                               # Dataset
│   ├── train/
│   │   ├── O/                                  # Organic images (training)
│   │   └── R/                                  # Recyclable images (training)
│   └── test/
│       ├── O/                                  # Organic images (test)
│       └── R/                                  # Recyclable images (test)
│
├── O_R_tlearn_vgg16.keras                      # Saved Extract Features model
├── O_R_tlearn_fine_tune_vgg16.keras           # Saved Fine-Tuned model
│
└── venv/                                       # Virtual environment (not versioned)
```

---

## 🚀 How to Use

### Running the Complete Notebook

**1. Open the Jupyter notebook**

```bash
jupyter notebook "Final Proj-Classify Waste Products Using TL- FT-v1.ipynb"
```

Or use VS Code / JupyterLab to open the `.ipynb` file

**2. Execute cells sequentially**

The notebook is organized into numbered tasks:

- **Task 1**: Check TensorFlow version
- **Task 2**: Create test data generator
- **Task 3**: Check training generator size
- **Task 4**: Visualize model summary
- **Task 5**: Compile the model
- **Task 6**: Plot accuracy curves (Extract Features)
- **Task 7**: Plot loss curves (Fine-Tuned)
- **Task 8**: Plot accuracy curves (Fine-Tuned)
- **Task 9**: Visualize predictions (Extract Features)
- **Task 10**: Visualize predictions (Fine-Tuned)

### Loading Pre-trained Models

```python
import tensorflow as tf

# Load Extract Features model
extract_feat_model = tf.keras.models.load_model('O_R_tlearn_vgg16.keras')

# Load Fine-Tuned model
fine_tune_model = tf.keras.models.load_model('O_R_tlearn_fine_tune_vgg16.keras')
```

### Making Predictions

```python
from tensorflow.keras.preprocessing import image
import numpy as np

# Load and preprocess image
img_path = 'path/to/waste/image.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make prediction
prediction = fine_tune_model.predict(img_array)
class_label = 'O' if prediction[0][0] < 0.5 else 'R'
print(f"Classification: {class_label}")
```

---

## 📈 Results

### Performance Metrics

The models were evaluated on the test set with 200 images (100 per class).

#### Extract Features Model

| Metric | Value |
|--------|-------|
| **Accuracy** | ~84-85% |
| **Validation** | Final accuracy of ~84.9% after 10 epochs |
| **Validation loss** | ~0.36 |

#### Fine-Tuned Model

| Metric | Value |
|--------|-------|
| **Accuracy** | Improved compared to Extract Features Model |
| **Validation** | Superior final accuracy after fine-tuning |
| **Validation loss** | Reduced compared to base model |

### Training Curves

The notebook includes visualizations of:

- **Loss Curves**: Training vs. Validation
- **Accuracy Curves**: Training vs. Validation
- **Model Comparison**: Extract Features vs. Fine-Tuned

### Prediction Visualizations

The project includes prediction visualizations on test images, showing:

- Original image
- True class
- Predicted class
- Confidence probability

---

## 🔬 Methodology

### 1. Data Preparation

- **Normalization**: Resizing to 150x150 pixels
- **Rescaling**: Pixel value normalization (0-255 → 0-1)
- **Data Augmentation** (training):
  - Horizontal flip (horizontal_flip)
  - Width shift (width_shift_range=0.1)
  - Height shift (height_shift_range=0.1)

### 2. Model Architecture

#### Extract Features Model

1. Load pre-trained VGG16 (ImageNet weights)
2. Freeze all convolutional layers
3. Add custom dense layers:
   - Flatten
   - Dense(512) + ReLU + Dropout(0.5)
   - Dense(512) + ReLU + Dropout(0.5)
   - Dense(1) + Sigmoid

#### Fine-Tuned Model

1. Unfreeze last convolutional layers of VGG16
2. Train with reduced learning rate
3. Keep trainable dense layers

### 3. Training

| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam |
| **Learning Rate** | 1e-4 initial with exponential decay |
| **Loss Function** | Binary Crossentropy |
| **Batch Size** | 32 |
| **Epochs** | 10 (with early stopping) |
| **Callbacks** | Early Stopping, Model Checkpoint, Learning Rate Scheduler |

**Callbacks used:**

- Early Stopping (patience=4, monitor='val_loss')
- Model Checkpoint (save best model)
- Learning Rate Scheduler (exponential decay)

### 4. Evaluation

- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Visualization**: Confusion matrix, classification reports
- **Test**: 200 images not seen during training

---

## 🤝 Contributions

Contributions are welcome! If you wish to contribute to this project:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Future Improvements

- [ ] Add more waste classes (paper, plastic, glass, etc.)
- [ ] Implement REST API for real-time predictions
- [ ] Create web interface for image upload
- [ ] Add support for real-time video
- [ ] Optimize model for mobile devices (TensorFlow Lite)
- [ ] Implement model ensemble
- [ ] Add prediction explanations (XAI)

---

## 📄 License

This project was developed as part of the **Deep Learning with Keras and TensorFlow** course on Coursera. The code is provided for educational purposes.

---

## 📚 References

### Articles and Documentation

- [VGG16 Paper](https://arxiv.org/abs/1409.1556) - Very Deep Convolutional Networks for Large-Scale Image Recognition
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning) - TensorFlow Official Documentation
- [Keras Applications](https://keras.io/api/applications/vgg/#vgg16-function) - VGG16 Pre-trained Model

### Datasets

- [Waste Classification Dataset](https://www.kaggle.com/datasets/techsash/waste-classification-data) - Kaggle Dataset

### Courses and Tutorials

- [Deep Learning with Keras and TensorFlow](https://www.coursera.org/) - Coursera Course
- [Skills Network](https://skills.network/) - IBM Skills Network

### Libraries

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)

---

## 👤 Author

**Sidnei Almeida**

- GitHub: [@sidnei-almeida](https://github.com/sidnei-almeida)
- LinkedIn: [Sidnei Almeida](https://www.linkedin.com/in/saaelmeida93/)
- Email: sidnei.almeida1806@gmail.com

---

## 🙏 Acknowledgments

- **Coursera** and **IBM Skills Network** for the excellent course
- **Kaggle** for providing the dataset
- Open-source community for the tools and libraries used

---

## 📧 Contact

For questions, suggestions, or collaborations, feel free to open an issue or get in touch.

---

<div align="center">

**If this project was useful to you, consider giving it a star! ⭐**

Made with ❤️ using TensorFlow and Keras

</div>
