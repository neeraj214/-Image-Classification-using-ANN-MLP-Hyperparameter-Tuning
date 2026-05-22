import os
import json

def load_json(file_name):
    """Safely loads a JSON file from the outputs directory."""
    path = os.path.join('outputs', file_name)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def main():
    print("Gathering benchmark data from outputs...")
    
    # 1. Load primary metadata
    mlp_baseline = load_json('mlp_baseline_meta.json')
    cnn_baseline = load_json('cnn_baseline_meta.json')
    mlp_grid = load_json('mlp_grid_meta.json')
    mlp_random = load_json('mlp_random_meta.json')
    
    # 2. Load comparison results
    activation_results = load_json('activation_comparison.json')
    optimizer_results = load_json('optimizer_comparison.json')

    # 3. Build comparison dictionary
    benchmark_dict = {}

    def add_to_benchmark(key, data):
        if data:
            benchmark_dict[key] = {
                "accuracy": data.get("accuracy") or data.get("test_accuracy"),
                "params": data.get("params") or data.get("params_count"),
                "training_time": data.get("training_time") or data.get("final_training_time")
            }

    add_to_benchmark("mlp_baseline", mlp_baseline)
    add_to_benchmark("cnn_baseline", cnn_baseline)
    add_to_benchmark("mlp_grid_tuned", mlp_grid)
    add_to_benchmark("mlp_random_tuned", mlp_random)

    # 4. Save benchmark results
    os.makedirs('outputs', exist_ok=True)
    save_path = os.path.join('outputs', 'benchmark.json')
    with open(save_path, 'w') as f:
        json.dump(benchmark_dict, f, indent=4)
    print(f"Benchmark data saved to {save_path}")

    # 5. Print formatted table
    print("\n" + "="*85)
    print(f"{'Model Name':<20} | {'Accuracy':<12} | {'Params':<15} | {'Training Time (s)':<20}")
    print("-" * 85)
    
    for model, stats in benchmark_dict.items():
        acc = f"{stats['accuracy']:.4f}" if stats['accuracy'] is not None else "N/A"
        params = f"{stats['params']:,}" if stats['params'] is not None else "N/A"
        time = f"{stats['training_time']:.2f}" if stats['training_time'] is not None else "N/A"
        
        print(f"{model:<20} | {acc:<12} | {params:<15} | {time:<20}")
    
    print("="*85)

    # 6. Additional insights from comparisons
    if activation_results:
        print("\nActivation Function Comparison (Accuracy):")
        for act, res in activation_results.items():
            print(f" - {act}: {res['accuracy']:.4f}")
            
    if optimizer_results:
        print("\nOptimizer Comparison (Accuracy):")
        for opt, res in optimizer_results.items():
            print(f" - {opt}: {res['accuracy']:.4f}")

if __name__ == '__main__':
    main()
