import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping

def build_model(optimizer_name='adam', learning_rate=0.001):
    """
    Builds an MLP model with a specific optimizer for comparison.
    """
    model = Sequential([
        Input(shape=(3072,)),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])
    
    if optimizer_name.lower() == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        opt = Adam(learning_rate=learning_rate)
        
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    # 1. Load preprocessed data
    print("Loading preprocessed CIFAR-10 data for optimizer comparison...")
    processed_dir = os.path.join('data', 'processed')
    X_train_flat = np.load(os.path.join(processed_dir, 'X_train_flat.npy'))
    X_test_flat = np.load(os.path.join(processed_dir, 'X_test_flat.npy'))
    y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))

    optimizers = ['adam', 'sgd', 'rmsprop']
    comparison_results = {}
    all_histories = {}

    os.makedirs('outputs', exist_ok=True)

    for opt_name in optimizers:
        print(f"\n--- Training MLP with {opt_name} optimizer ---")
        model = build_model(optimizer_name=opt_name)
        
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
        print(f"Test Accuracy ({opt_name}): {test_acc:.4f}")
        
        comparison_results[opt_name] = {
            "accuracy": float(test_acc),
            "time": float(training_time)
        }
        all_histories[opt_name] = history.history['val_accuracy']

    # 2. Save JSON results
    results_path = os.path.join('outputs', 'optimizer_comparison.json')
    with open(results_path, 'w') as f:
        json.dump(comparison_results, f, indent=4)
    print(f"\nComparison results saved to {results_path}")

    # 3. Plot validation accuracy curves
    plt.figure(figsize=(10, 6))
    for opt_name, val_acc_curve in all_histories.items():
        plt.plot(val_acc_curve, label=f'Optimizer: {opt_name}')
    
    plt.title('MLP Validation Accuracy by Optimizer')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join('outputs', 'optimizer_comparison.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Comparison plot saved to {plot_path}")

if __name__ == '__main__':
    main()
