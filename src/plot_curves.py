import os
import json
import matplotlib.pyplot as plt

def load_history(file_name):
    """Loads history dictionary from a JSON file in the outputs directory."""
    path = os.path.join('outputs', file_name)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def plot_subplot(ax, history, title, metric='accuracy'):
    """Plots training and validation curves for a specific metric on a given axis."""
    if not history:
        ax.text(0.5, 0.5, f'History not found', ha='center', va='center')
        ax.set_title(title)
        return

    epochs = range(1, len(history[metric]) + 1)
    ax.plot(epochs, history[metric], 'bo-', label=f'Training {metric}')
    
    val_metric = f'val_{metric}'
    if val_metric in history:
        ax.plot(epochs, history[val_metric], 'ro-', label=f'Validation {metric}')
    
    ax.set_title(title)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(metric.capitalize())
    ax.legend()
    ax.grid(True)

def main():
    # 1. Load baseline histories
    mlp_history = load_history('mlp_baseline_history.json')
    cnn_history = load_history('cnn_baseline_history.json')
    
    # 2. Setup 2x2 Subplots for Baselines
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # MLP Curves
    plot_subplot(axes[0, 0], mlp_history, 'MLP Baseline: Accuracy', 'accuracy')
    plot_subplot(axes[1, 0], mlp_history, 'MLP Baseline: Loss', 'loss')
    
    # CNN Curves
    plot_subplot(axes[0, 1], cnn_history, 'CNN Baseline: Accuracy', 'accuracy')
    plot_subplot(axes[1, 1], cnn_history, 'CNN Baseline: Loss', 'loss')
    
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    save_path = os.path.join('outputs', 'training_curves.png')
    plt.savefig(save_path)
    print(f"Main training curves saved to {save_path}")
    
    # 3. Optional: Plot Tuning Histories (Grid/Random) if available
    # Note: These usually don't have separate history JSONs in the requested tasks, 
    # but we check for common naming patterns.
    tuning_histories = {
        'MLP Grid Search': load_history('mlp_grid_history.json'),
        'MLP Random Search': load_history('mlp_random_history.json')
    }
    
    available_tuning = {k: v for k, v in tuning_histories.items() if v is not None}
    
    if available_tuning:
        fig_tune, axes_tune = plt.subplots(len(available_tuning), 2, figsize=(15, 6 * len(available_tuning)))
        if len(available_tuning) == 1:
            axes_tune = [axes_tune]
            
        for i, (name, hist) in enumerate(available_tuning.items()):
            plot_subplot(axes_tune[i][0], hist, f'{name}: Accuracy', 'accuracy')
            plot_subplot(axes_tune[i][1], hist, f'{name}: Loss', 'loss')
            
        plt.tight_layout()
        tune_save_path = os.path.join('outputs', 'tuning_curves.png')
        plt.savefig(tune_save_path)
        print(f"Tuning curves saved to {tune_save_path}")

if __name__ == '__main__':
    main()
