import os
import time
import json
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

def build_mlp(neurons=512, dropout=0.3, learning_rate=0.001):
    """
    Builds and compiles an MLP model for CIFAR-10 classification.
    
    Architecture:
    Input(3072) -> Dense(neurons, relu) -> Dropout -> Dense(neurons//2, relu) -> 
    Dropout -> Dense(neurons//4, relu) -> Dropout -> Dense(10, softmax)
    """
    model = Sequential([
        Input(shape=(3072,)),
        Dense(neurons, activation='relu'),
        Dropout(dropout),
        Dense(neurons // 2, activation='relu'),
        Dropout(dropout),
        Dense(neurons // 4, activation='relu'),
        Dropout(dropout),
        Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    # 1. Load preprocessed data
    print("Loading preprocessed CIFAR-10 data...")
    processed_dir = os.path.join('data', 'processed')
    X_train_flat = np.load(os.path.join(processed_dir, 'X_train_flat.npy'))
    X_test_flat = np.load(os.path.join(processed_dir, 'X_test_flat.npy'))
    y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))
    
    print(f"X_train_flat shape: {X_train_flat.shape}")
    print(f"X_test_flat shape: {X_test_flat.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # 2. Build model
    print("\nBuilding MLP baseline model...")
    model = build_mlp(neurons=512, dropout=0.3, learning_rate=0.001)
    model.summary()
    
    # 3. Define EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    
    # 4. Train model and record training time
    print("\nStarting model training...")
    start_time = time.time()
    
    history = model.fit(
        X_train_flat,
        y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")
    
    # Ensure save directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # 5. Save model
    model_save_path = os.path.join('models', 'mlp_baseline.h5')
    print(f"Saving model to {model_save_path}...")
    model.save(model_save_path)
    
    # 6. Save history (convert values to standard float for JSON serialization)
    history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
    history_save_path = os.path.join('outputs', 'mlp_baseline_history.json')
    print(f"Saving training history to {history_save_path}...")
    with open(history_save_path, 'w') as f:
        json.dump(history_dict, f, indent=4)
        
    # 7. Evaluate model on test data
    print("\nEvaluating model on test data...")
    test_loss, test_acc = model.evaluate(X_test_flat, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # 8. Generate classification report
    y_pred = model.predict(X_test_flat)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print("\nClassification Report:")
    report = classification_report(y_test_classes, y_pred_classes, target_names=class_names)
    print(report)
    
    # 9. Save metadata
    num_params = model.count_params()
    metadata = {
        "accuracy": float(test_acc),
        "params": int(num_params),
        "training_time": float(training_time)
    }
    metadata_save_path = os.path.join('outputs', 'mlp_baseline_meta.json')
    print(f"Saving metadata to {metadata_save_path}...")
    with open(metadata_save_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("\nBaseline MLP script completed successfully.")

if __name__ == '__main__':
    main()
