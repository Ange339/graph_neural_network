import itertools
import yaml
import subprocess
import os

# Define the hyperparameter search space
search_space = {
    "learning_rate": [0.001, 0.01, 0.1],
    "batch_size": [32, 64, 128],
    "hidden_channels": [64, 128, 256],
}

combinations = list(itertools.product(*search_space.values()))

# Path to the config file and train script
config_path = "config.yaml"
train_script = "train.py"

# Directory to save results
results_dir = "grid_search_results"
os.makedirs(results_dir, exist_ok=True)

# Iterate over all combinations
for i, combo in enumerate(combinations):
    # Create a dictionary for the current configuration
    current_config = dict(zip(search_space.keys(), combo))
    
    # Load the existing config file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Update the config with the current hyperparameters
    config.update(current_config)
    
    # Save the updated config to a temporary file
    temp_config_path = f"temp_config_{i}.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f)
    
    # Run the training script with the updated config
    print(f"Running configuration {i + 1}/{len(combinations)}: {current_config}")
    subprocess.run(["python", train_script, "--config", temp_config_path])
    
    # Move the training log or results to the results directory
    os.rename("train.log", os.path.join(results_dir, f"train_{i}.log"))
    
    # Remove the temporary config file
    os.remove(temp_config_path)