import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10

def main():
    print("Loading CIFAR-10 dataset...")
    # Load CIFAR-10 using keras.datasets.cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Print dataset shape: train (50000, 32, 32, 3), test (10000, 32, 32, 3)
    print("\n--- Dataset Shapes ---")
    print(f"Train features shape: {x_train.shape}")
    print(f"Train labels shape: {y_train.shape}")
    print(f"Test features shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")

    # Print class names: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print("\n--- Class Names ---")
    print(", ".join(class_names))

    # Print pixel value range before normalization
    min_val = x_train.min()
    max_val = x_train.max()
    print("\n--- Pixel Value Range (Before Normalization) ---")
    print(f"Min pixel value: {min_val}")
    print(f"Max pixel value: {max_val}")

    # Print class distribution (should be balanced: 5000 per class)
    print("\n--- Class Distribution in Training Set ---")
    unique, counts = np.unique(y_train, return_counts=True)
    for class_idx, count in zip(unique, counts):
        print(f"Class {class_idx} ({class_names[class_idx]}): {count} samples")

    # Plot 5x10 sample grid (one row per class, 5 samples each), save as outputs/sample_grid.png
    print("\nGenerating sample grid plot...")
    
    # 10 rows (classes) by 5 columns (samples)
    fig, axes = plt.subplots(10, 5, figsize=(10, 15))
    
    for class_idx in range(10):
        # Get indices of training images belonging to this class
        indices = np.where(y_train.flatten() == class_idx)[0]
        # Take the first 5 samples
        selected_indices = indices[:5]
        
        for col_idx, img_idx in enumerate(selected_indices):
            ax = axes[class_idx, col_idx]
            ax.imshow(x_train[img_idx])
            ax.axis('off')
            
            # Label the row on the first column
            if col_idx == 0:
                ax.text(-10, 16, class_names[class_idx], 
                        va='center', ha='right', fontsize=12, fontweight='bold')
                
    plt.tight_layout()
    
    # Ensure outputs directory exists
    os.makedirs('outputs', exist_ok=True)
    output_path = os.path.join('outputs', 'sample_grid.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Sample grid successfully saved to {output_path}")

if __name__ == '__main__':
    main()
