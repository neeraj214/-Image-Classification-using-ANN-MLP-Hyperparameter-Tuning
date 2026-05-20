import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

def build_comparison_mlp(activation='relu', neurons=512, dropout=0.3, learning_rate=0.001):
    """
    Builds an MLP with a specific activation function for comparison.
    """
    model = Sequential([
        Input(shape=(3072,)),
        Dense(neurons, activation=activation),
        Dropout(dropout),
        Dense(neurons // 2, activation=activation),
        Dropout(dropout),
        Dense(neurons // 4, activation=activation),
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
    print("Loading preprocessed CIFAR-10 data for comparison...")
    processed_dir = os.path.join('data', 'processed')
    X_train_flat = np.load(os.path.join(processed_dir, 'X_train_flat.npy'))
    X_test_flat = np.load(os.path.join(processed_dir, 'X_test_flat.npy'))
    y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))

    activations = ['relu', 'sigmoid', 'tanh']
    comparison_results = {}
    all_histories = {}

    os.makedirs('outputs', exist_ok=True)

    for act in activations:
        print(f"\n--- Training MLP with {act} activation ---")
        model = build_comparison_mlp(activation=act)
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        start_time = time.time()
        history = model.fit(
            X_train_flat,
            y_train,
            epochs=30,
            batch_size=64,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=1
        )
        training_time = time.time() - start_time
        
        # Evaluate
        _, test_acc = model.evaluate(X_test_flat, y_test, verbose=0)
        print(f"Test Accuracy ({act}): {test_acc:.4f}")
        
        comparison_results[act] = {
            "accuracy": float(test_acc),
            "time": float(training_time)
        }
        all_histories[act] = history.history['accuracy']

    # 2. Save JSON results
    results_path = os.path.join('outputs', 'activation_comparison.json')
    with open(results_path, 'w') as f:
        json.dump(comparison_results, f, indent=4)
    print(f"\nComparison results saved to {results_path}")

    # 3. Plot accuracy curves
    plt.figure(figsize=(10, 6))
    for act, acc_curve in all_histories.items():
        plt.plot(acc_curve, label=f'Activation: {act}')
    
    plt.title('MLP Training Accuracy by Activation Function')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join('outputs', 'activation_comparison.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Comparison plot saved to {plot_path}")

if __name__ == '__main__':
    main()
