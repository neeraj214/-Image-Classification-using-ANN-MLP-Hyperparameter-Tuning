import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

def main():
    # 1. Setup paths and class names
    processed_dir = os.path.join('data', 'processed')
    models_dir = 'models'
    outputs_dir = 'outputs'
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 2. Load test data
    print("Loading test data...")
    X_test_flat = np.load(os.path.join(processed_dir, 'X_test_flat.npy'))
    X_test_cnn = np.load(os.path.join(processed_dir, 'X_test_cnn.npy'))
    y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))
    y_test_classes = np.argmax(y_test, axis=1)

    # 3. Load models
    print("Loading models...")
    mlp_model = load_model(os.path.join(models_dir, 'mlp_baseline.h5'))
    cnn_model = load_model(os.path.join(models_dir, 'cnn_baseline.h5'))

    # 4. Generate predictions
    print("Generating predictions...")
    y_pred_mlp = np.argmax(mlp_model.predict(X_test_flat), axis=1)
    y_pred_cnn = np.argmax(cnn_model.predict(X_test_cnn), axis=1)

    # 5. Print classification reports
    print("\n" + "="*30)
    print("MLP CLASSIFICATION REPORT")
    print("="*30)
    print(classification_report(y_test_classes, y_pred_mlp, target_names=class_names))

    print("\n" + "="*30)
    print("CNN CLASSIFICATION REPORT")
    print("="*30)
    print(classification_report(y_test_classes, y_pred_cnn, target_names=class_names))

    # 6. Plot Confusion Matrices
    print("\nPlotting confusion matrices...")
    cm_mlp = confusion_matrix(y_test_classes, y_pred_mlp)
    cm_cnn = confusion_matrix(y_test_classes, y_pred_cnn)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # MLP Heatmap
    sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=class_names, yticklabels=class_names)
    ax1.set_title('MLP Baseline Confusion Matrix')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')

    # CNN Heatmap
    sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Greens', ax=ax2,
                xticklabels=class_names, yticklabels=class_names)
    ax2.set_title('CNN Baseline Confusion Matrix')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')

    plt.tight_layout()
    
    # Save the plot
    os.makedirs(outputs_dir, exist_ok=True)
    save_path = os.path.join(outputs_dir, 'confusion_matrices.png')
    plt.savefig(save_path)
    print(f"Confusion matrices saved to {save_path}")

if __name__ == '__main__':
    main()
