import os
import json
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

app = FastAPI(title="CIFAR-10 Visual Classifier API")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and class names
cnn_model = None
mlp_model = None
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@app.on_event("startup")
async def startup_event():
    """Load models and class names on startup."""
    global cnn_model, mlp_model
    try:
        cnn_model_path = os.path.join("models", "cnn_baseline.h5")
        mlp_model_path = os.path.join("models", "mlp_baseline.h5")
        
        if os.path.exists(cnn_model_path):
            cnn_model = load_model(cnn_model_path)
            print(f"CNN model loaded from {cnn_model_path}")
        else:
            print(f"Warning: CNN model not found at {cnn_model_path}")

        if os.path.exists(mlp_model_path):
            mlp_model = load_model(mlp_model_path)
            print(f"MLP model loaded from {mlp_model_path}")
        else:
            print(f"Warning: MLP model not found at {mlp_model_path}")
            
    except Exception as e:
        print(f"Error during model loading: {e}")

@app.get("/health")
async def health():
    """Returns the API health status and loaded models."""
    models_loaded = []
    if cnn_model: models_loaded.append("cnn")
    if mlp_model: models_loaded.append("mlp")
    return {"status": "ok", "models": models_loaded}

@app.get("/benchmark")
async def get_benchmark():
    """Reads and returns the benchmark results from outputs/benchmark.json."""
    benchmark_path = os.path.join("outputs", "benchmark.json")
    if os.path.exists(benchmark_path):
        with open(benchmark_path, "r") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Benchmark results not found")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives an image file, processes it, and returns predictions 
    from both CNN and MLP models.
    """
    if not cnn_model or not mlp_model:
        raise HTTPException(status_code=503, detail="Models are not loaded on the server")

    try:
        # 1. Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((32, 32))
        
        # 2. Normalize and prepare for models
        img_array = np.array(image).astype("float32") / 255.0
        
        # CNN Input: (1, 32, 32, 3)
        cnn_input = np.expand_dims(img_array, axis=0)
        
        # MLP Input: (1, 3072)
        mlp_input = img_array.reshape(1, -1)

        # 3. Run predictions
        cnn_preds = cnn_model.predict(cnn_input, verbose=0)[0]
        mlp_preds = mlp_model.predict(mlp_input, verbose=0)[0]

        # 4. Format output
        def format_result(preds):
            class_idx = np.argmax(preds)
            return {
                "class": class_names[class_idx],
                "confidence": float(preds[class_idx]),
                "probabilities": {name: float(prob) for name, prob in zip(class_names, preds)}
            }

        return {
            "cnn": format_result(cnn_preds),
            "mlp": format_result(mlp_preds)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
