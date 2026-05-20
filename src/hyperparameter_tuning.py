import os
import json
import numpy as np
from src.mlp_model import build_mlp
from keras.callbacks import EarlyStopping

def main():
    # 1. Load preprocessed data
    print("Loading preprocessed CIFAR-10 data for tuning...")
    processed_dir = os.path.join('data', 'processed')
    X_train_flat = np.load(os.path.join(processed_dir, 'X_train_flat.npy'))
    X_test_flat = np.load(os.path.join(processed_dir, 'X_test_flat.npy'))
    y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))

    # Define hyperparameter grid
    param_grid = {
        'neurons': [256, 512],
        'dropout': [0.2, 0.4],
        'learning_rate': [0.001, 0.0001]
    }

    best_acc = 0
    best_params = {}
    results = []

    print("\nStarting Hyperparameter Tuning (Simple Grid Search)...")
    
    for neurons in param_grid['neurons']:
        for dropout in param_grid['dropout']:
            for lr in param_grid['learning_rate']:
                print(f"\nTesting: neurons={neurons}, dropout={dropout}, learning_rate={lr}")
                
                model = build_mlp(neurons=neurons, dropout=dropout, learning_rate=lr)
                
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )
                
                history = model.fit(
                    X_train_flat,
                    y_train,
                    epochs=10, # Reduced epochs for tuning speed
                    batch_size=128,
                    validation_split=0.1,
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                val_acc = max(history.history['val_accuracy'])
                print(f"Validation Accuracy: {val_acc:.4f}")
                
                result = {
                    'params': {'neurons': neurons, 'dropout': dropout, 'learning_rate': lr},
                    'val_accuracy': float(val_acc)
                }
                results.append(result)
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_params = result['params']

    print(f"\nBest Validation Accuracy: {best_acc:.4f}")
    print(f"Best Parameters: {best_params}")

    # Save results
    os.makedirs('outputs', exist_ok=True)
    tuning_save_path = os.path.join('outputs', 'tuning_results.json')
    with open(tuning_save_path, 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_val_accuracy': float(best_acc),
            'all_results': results
        }, f, indent=4)
    
    print(f"Tuning results saved to {tuning_save_path}")

if __name__ == '__main__':
    main()
