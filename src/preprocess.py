import os
import numpy as np
import keras
from keras.datasets import cifar10
from keras.utils import to_categorical

def main():
    print("Loading CIFAR-10 dataset...")
    # Load CIFAR-10 via keras
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    print("Normalizing image pixel values...")
    # Normalize images: divide by 255.0 -> float32 in [0,1]
    x_train_norm = x_train.astype('float32') / 255.0
    x_test_norm = x_test.astype('float32') / 255.0

    print("One-hot encoding labels...")
    # One-hot encode labels using to_categorical(y, num_classes=10)
    y_train_encoded = to_categorical(y_train, num_classes=10)
    y_test_encoded = to_categorical(y_test, num_classes=10)

    print("Preparing MLP flattened features...")
    # For MLP: flatten images -> shape (50000, 3072)
    x_train_flat = x_train_norm.reshape(x_train_norm.shape[0], -1)
    x_test_flat = x_test_norm.reshape(x_test_norm.shape[0], -1)

    print("Preparing CNN features...")
    # For CNN: keep shape (50000, 32, 32, 3)
    x_train_cnn = x_train_norm
    x_test_cnn = x_test_norm

    # Ensure target directory exists
    processed_dir = os.path.join('data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    print("Saving processed numpy arrays...")
    # Save as numpy arrays in data/processed/
    np.save(os.path.join(processed_dir, 'X_train_flat.npy'), x_train_flat)
    np.save(os.path.join(processed_dir, 'X_test_flat.npy'), x_test_flat)
    np.save(os.path.join(processed_dir, 'X_train_cnn.npy'), x_train_cnn)
    np.save(os.path.join(processed_dir, 'X_test_cnn.npy'), x_test_cnn)
    np.save(os.path.join(processed_dir, 'y_train.npy'), y_train_encoded)
    np.save(os.path.join(processed_dir, 'y_test.npy'), y_test_encoded)

    # Print saved shapes confirmation
    print("\n--- Saved Shapes Confirmation ---")
    print(f"MLP Train features (X_train_flat): {x_train_flat.shape}")
    print(f"MLP Test features (X_test_flat): {x_test_flat.shape}")
    print(f"CNN Train features (X_train_cnn): {x_train_cnn.shape}")
    print(f"CNN Test features (X_test_cnn): {x_test_cnn.shape}")
    print(f"Train labels (y_train): {y_train_encoded.shape}")
    print(f"Test labels (y_test): {y_test_encoded.shape}")
    print("\nPreprocessing completed successfully.")

if __name__ == '__main__':
    main()
