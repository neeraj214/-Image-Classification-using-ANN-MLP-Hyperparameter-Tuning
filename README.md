# CIFAR-10 Image Classification: MLP vs CNN

This project provides a comprehensive pipeline for image classification on the CIFAR-10 dataset. It benchmarks a standard Multi-Layer Perceptron (MLP) against a Convolutional Neural Network (CNN), including hyperparameter tuning and activation function analysis.

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Workflow](#workflow)
- [Model Architectures](#model-architectures)
- [Results & Outputs](#results--outputs)

## 🚀 Project Overview
The goal of this project is to demonstrate the performance difference between dense neural networks (ANN/MLP) and spatial-aware networks (CNN) for image recognition. It covers:
- Data exploration and visualization.
- Data preprocessing (normalization, flattening, one-hot encoding).
- Performance benchmarking of different activation functions.
- Automated hyperparameter tuning using grid search.

## 📂 Project Structure
```text
├── data/               # Processed datasets (numpy arrays)
├── models/             # Saved trained models (.h5)
├── outputs/            # Plots, logs, and evaluation metrics
├── src/                # Source code
│   ├── __init__.py
│   ├── data_exploration.py
│   ├── preprocess.py
│   ├── mlp_model.py
│   ├── cnn_model.py
│   ├── activation_comparison.py
│   └── hyperparameter_tuning.py
└── README.md
```

## 🛠 Requirements
Install the necessary libraries using pip:
```bash
pip install numpy matplotlib keras tensorflow scikit-learn
```

## 🔄 Workflow

### 1. Exploration
Visualize the CIFAR-10 classes and data distribution.
```bash
python src/data_exploration.py
```

### 2. Preprocessing
Prepare the data for both MLP (flattened) and CNN (3D) models.
```bash
python src/preprocess.py
```

### 3. Training
Train the baseline models to establish performance benchmarks.
```bash
python src/mlp_model.py
python src/cnn_model.py
```

### 4. Analysis
Compare activation functions (ReLU, Sigmoid, Tanh) or run hyperparameter tuning.
```bash
python src/activation_comparison.py
python src/hyperparameter_tuning.py
```

## 🧠 Model Architectures

### Multi-Layer Perceptron (MLP)
- **Input**: Flattened 3072 features (32x32x3).
- **Hidden Layers**: Dense(512) → Dense(256) → Dense(128).
- **Regularization**: Dropout (0.3) after each hidden layer.

### Convolutional Neural Network (CNN)
- **Feature Extraction**: 4x Conv2D layers with BatchNormalization.
- **Downsampling**: MaxPooling2D after every two conv layers.
- **Classifier**: Flatten → Dense(256) → Dense(10).

## 📊 Results & Outputs
- **Plots**: `outputs/sample_grid.png`, `outputs/activation_comparison.png`.
- **Metrics**: `outputs/*_meta.json` (Accuracy, training time, params).
- **History**: `outputs/*_history.json` (Epoch-wise loss and accuracy).
- **Models**: `models/*.h5` (Pre-trained model weights).
