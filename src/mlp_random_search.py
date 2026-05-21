import os
import json
import time
import numpy as np
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from src.mlp_model import build_mlp

class KerasMLPWrapper(BaseEstimator, ClassifierMixin):
    """
    Manual wrapper for Keras MLP model to work with sklearn RandomizedSearchCV
    without requiring scikeras or the deprecated keras.wrappers.
    """
    def __init__(self, neurons=512, dropout=0.3, learning_rate=0.001, epochs=30, batch_size=64):
        self.neurons = neurons
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.classes_ = np.arange(10)

    def fit(self, X, y):
        self.model = build_mlp(
            neurons=int(self.neurons), 
            dropout=float(self.dropout), 
            learning_rate=float(self.learning_rate)
        )
        self.model.fit(
            X, y, 
            epochs=int(self.epochs), 
            batch_size=int(self.batch_size), 
            verbose=0
        )
        return self

    def predict(self, X):
        y_pred = self.model.predict(X, verbose=0)
        return np.argmax(y_pred, axis=1)

    def score(self, X, y):
        y_true_classes = np.argmax(y, axis=1)
        y_pred_classes = self.predict(X)
        return np.mean(y_true_classes == y_pred_classes)

def main():
    # 1. Load data
    print("Loading data for Random Search...")
    processed_dir = os.path.join('data', 'processed')
    X_train = np.load(os.path.join(processed_dir, 'X_train_flat.npy'))
    y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(processed_dir, 'X_test_flat.npy'))
    y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))

    # 2. Define Parameter Distribution
    param_distributions = {
        'neurons': [256, 512, 1024],
        'dropout': uniform(0.1, 0.4), # uniform(0.1, 0.5) implies loc=0.1, scale=0.4
        'learning_rate': uniform(0.0001, 0.0099), # uniform(0.0001, 0.01) implies loc=0.0001, scale=0.0099
        'batch_size': [32, 64],
        'epochs': [30]
    }

    # 3. Setup RandomizedSearchCV
    print("\nStarting Random Search (n_iter=8, cv=3)...")
    base_model = KerasMLPWrapper()
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=8,
        cv=3,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1,
        verbose=2
    )

    start_time = time.time()
    search_result = random_search.fit(X_train, y_train)
    total_time = time.time() - start_time

    # 4. Results
    print(f"\nBest Score: {search_result.best_score_:.4f}")
    print(f"Best Params: {search_result.best_params_}")

    # 5. Save best params
    os.makedirs('outputs', exist_ok=True)
    with open(os.path.join('outputs', 'mlp_random_best_params.json'), 'w') as f:
        json.dump(search_result.best_params_, f, indent=4)

    # 6. Retrain best model on full train set
    print("\nRetraining best model on full training set...")
    best_params = search_result.best_params_
    final_model = build_mlp(
        neurons=int(best_params['neurons']),
        dropout=float(best_params['dropout']),
        learning_rate=float(best_params['learning_rate'])
    )
    
    final_start_time = time.time()
    final_model.fit(
        X_train, y_train,
        epochs=int(best_params['epochs']),
        batch_size=int(best_params['batch_size']),
        verbose=1
    )
    final_train_time = time.time() - final_start_time

    # 7. Evaluate and Save
    os.makedirs('models', exist_ok=True)
    final_model.save(os.path.join('models', 'mlp_random_best.h5'))
    
    test_loss, test_acc = final_model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    metadata = {
        "best_score": float(search_result.best_score_),
        "best_params": best_params,
        "test_accuracy": float(test_acc),
        "total_search_time": float(total_time),
        "final_training_time": float(final_train_time),
        "params_count": int(final_model.count_params())
    }
    
    with open(os.path.join('outputs', 'mlp_random_meta.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("\nRandom Search script completed successfully.")

if __name__ == '__main__':
    main()
