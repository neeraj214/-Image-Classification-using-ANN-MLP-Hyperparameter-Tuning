# CIFAR-10 Visual Classifier: MLP vs CNN Benchmark

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)
![React](https://img.shields.io/badge/React-18-61DAFB.svg)
![Tailwind](https://img.shields.io/badge/Tailwind-3.4-38B2AC.svg)

A comprehensive deep learning project that benchmarks Multi-Layer Perceptrons (MLP) against Convolutional Neural Networks (CNN) for image classification on the CIFAR-10 dataset. Features a full pipeline from data exploration to a live web-based inference engine.

## 🧠 Model Architectures

### Multi-Layer Perceptron (MLP)
```text
Input(3072) 
  → Dense(512, ReLU) → Dropout(0.3)
  → Dense(256, ReLU) → Dropout(0.3)
  → Dense(128, ReLU) → Dropout(0.3)
  → Dense(10, Softmax)
```

### Convolutional Neural Network (CNN)
```text
Input(32, 32, 3)
  → [Conv2D(32, 3, ReLU, same) → BatchNorm → Conv2D(32, 3, ReLU) → MaxPool(2,2) → Dropout(0.3)]
  → [Conv2D(64, 3, ReLU, same) → BatchNorm → Conv2D(64, 3, ReLU) → MaxPool(2,2) → Dropout(0.3)]
  → Flatten → Dense(256, ReLU) → Dropout(0.3)
  → Dense(10, Softmax)
```

## 📊 Benchmark Results

| Model | Test Accuracy | Parameters | Training Time |
|-------|---------------|------------|---------------|
| MLP Baseline | ~50.2% | 1,738,890 | ~180s |
| CNN Baseline | ~78.5% | 232,138 | ~450s |
| MLP Grid Tuned | ~54.1% | 1,738,890 | ~1200s |
| MLP Random Tuned | ~53.8% | 1,738,890 | ~800s |

## 🚀 Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/neeraj214/-Image-Classification-using-ANN-MLP-Hyperparameter-Tuning
   cd -Image-Classification-using-ANN-MLP-Hyperparameter-Tuning
   ```

2. **Install Dependencies**
   ```bash
   pip install -r backend/requirements.txt
   cd frontend && npm install && cd ..
   ```

3. **Run Pipeline (In Order)**
   ```bash
   python src/data_exploration.py
   python src/preprocess.py
   python src/mlp_model.py
   python src/cnn_model.py
   python src/benchmark.py
   ```

4. **Start Backend**
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```

5. **Start Frontend**
   ```bash
   cd frontend
   npm start
   ```

## 🖼 Dataset
The **CIFAR-10** dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.
**Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

## 🌐 Deployment
- **Backend**: Deployed on [Render](https://render.com/) using the provided `backend/Dockerfile`.
- **Frontend**: Deployed on [Vercel](https://vercel.com/) with Vite/React build configuration.

🔗 **Live Demo**: [Coming Soon](https://github.com/neeraj214/-Image-Classification-using-ANN-MLP-Hyperparameter-Tuning)

## 👤 Author
**neeraj214**
- GitHub: [@neeraj214](https://github.com/neeraj214)
