import os
import time
import json
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

def build_cnn(filters=32, dropout=0.3, learning_rate=0.001):
    """
    Builds and compiles a CNN model for CIFAR-10 classification.
    
    Architecture:
    Conv2D(filters, 3, relu, same) -> BatchNorm -> Conv2D(filters, 3, relu) -> MaxPool(2,2) -> Dropout
    -> Conv2D(filters*2, 3, relu, same) -> BatchNorm -> Conv2D(filters*2, 3, relu) -> MaxPool(2,2) -> Dropout
    -> Flatten -> Dense(256, relu) -> Dropout -> Dense(10, softmax)
    """
    model = Sequential([
        Input(shape=(32, 32, 3)),
        
        # First Block
        Conv2D(filters, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(filters, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout),
        
        # Second Block
        Conv2D(filters * 2, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(filters * 2, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout),
        
        # Fully Connected Block
        Flatten(),
        Dense(256, activation='relu'),
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
    # 1. Load preprocessed CNN data
    print("Loading preprocessed CIFAR-10 CNN data...")
    processed_dir = os.path.join('data', 'processed')
    X_train_cnn = np.load(os.path.join(processed_dir, 'X_train_cnn.npy'))
    X_test_cnn = np.load(os.path.join(processed_dir, 'X_test_cnn.npy'))
    y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))
    
    print(f"X_train_cnn shape: {X_train_cnn.shape}")
    print(f"X_test_cnn shape: {X_test_cnn.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # 2. Build model
    print("\nBuilding CNN baseline model...")
    model = build_cnn(filters=32, dropout=0.3, learning_rate=0.001)
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
        X_train_cnn,
        y_train,
        epochs=5,
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
    model_save_path = os.path.join('models', 'cnn_baseline.h5')
    print(f"Saving model to {model_save_path}...")
    model.save(model_save_path)
    
    # 6. Save history
    history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
    history_save_path = os.path.join('outputs', 'cnn_baseline_history.json')
    print(f"Saving training history to {history_save_path}...")
    with open(history_save_path, 'w') as f:
        json.dump(history_dict, f, indent=4)
        
    # 7. Evaluate model on test data
    print("\nEvaluating model on test data...")
    test_loss, test_acc = model.evaluate(X_test_cnn, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # 8. Generate classification report
    y_pred = model.predict(X_test_cnn)
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
    metadata_save_path = os.path.join('outputs', 'cnn_baseline_meta.json')
    print(f"Saving metadata to {metadata_save_path}...")
    with open(metadata_save_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("\nBaseline CNN script completed successfully.")

if __name__ == '__main__':
    main()
