# CIFAR-10 Image Classification using ANN/MLP

This project implements a complete pipeline for classifying images from the CIFAR-10 dataset using an Artificial Neural Network (Multi-Layer Perceptron). It includes data exploration, preprocessing, model training, and hyperparameter tuning.

## Project Structure

- `src/`: Source code for the project.
  - `data_exploration.py`: Loads the dataset and generates a sample grid of images.
  - `preprocess.py`: Normalizes images and prepares them for the MLP model.
  - `mlp_model.py`: Builds, trains, and evaluates the baseline MLP model.
  - `cnn_model.py`: Builds, trains, and evaluates the baseline CNN model.
  - `hyperparameter_tuning.py`: Performs a grid search to find the best model parameters.
- `data/`: (Generated) Contains processed numpy arrays.
- `models/`: (Generated) Contains saved model files (`.h5`).
- `outputs/`: (Generated) Contains plots, training history, and evaluation metrics.

## Setup and Usage

1. **Install Dependencies**:
   ```bash
   pip install numpy matplotlib keras tensorflow scikit-learn
   ```

2. **Data Exploration**:
   Run the exploration script to see dataset details and a sample grid.
   ```bash
   python src/data_exploration.py
   ```

3. **Preprocessing**:
   Prepare the data for training.
   ```bash
   python src/preprocess.py
   ```

4. **Training Baseline Models**:
   Train the baseline MLP model:
   ```bash
   python src/mlp_model.py
   ```
   Train the baseline CNN model:
   ```bash
   python src/cnn_model.py
   ```

5. **Hyperparameter Tuning**:
   Run the tuning script to find optimal parameters.
   ```bash
   python src/hyperparameter_tuning.py
   ```

## Results

- Training history and classification reports are saved in the `outputs/` directory.
- The best model is saved in the `models/` directory.
