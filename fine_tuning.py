import os
import random
import yaml
import subprocess
import copy
from itertools import product
from datetime import datetime

# Define the search space for hyperparameters
search_space = {
    'user_features': [
        [{'name': 'textual_reviews', 'aggr_fn': 'mean'}],
        [{'name': 'textual_reviews', 'aggr_fn': 'sum'}],
    ],
    'book_features': [
        [{'name': 'textual_desc'}, {'name': 'textual_reviews', 'aggr_fn': 'mean'}],
        [{'name': 'textual_desc'}, {'name': 'textual_reviews', 'aggr_fn': 'sum'}],
        # Add more variants as needed
    ],
    "hidden_channels": [32, 128, 512, 1024, 2048],
    "latent_dim": [32, 128, 512, 1024, 2048],
    'num_layers': [1, 2, 3],
    'user_emb_dim': [32, 128, 512, 1024, 2048],
    'book_emb_dim': [32, 128, 512, 1024, 2048],
    'user_feature_linear': [True, False],
    'book_feature_linear': [True, False],
    "skip_connection": ["sum", "concat", None],
    "batch_norm": [True, False],
    'dropout': [0.0, 0.2, 0.5],
    "heads": [1, 2, 5],
    "encoder": ["sage_encoder", "gat_encoder"],
    "variational": [True, False],
    "negative_sampling_method": ["batch_random", "pairwise_random"],
    "recon_loss": ["bpr", "binary"],
    'learning_rate': [0.001, 0.0005, 0.0001],
    'kl_beta': [0.1, 0.2, 1.0],
    'kl_warmup_epoch': [0, 5, 10],
}

CONFIG_PATH = 'config.yaml'
TRAIN_SCRIPT = 'train.py'
OUT_DIR = 'search_results'  # Directory to save results
LOG_FILE = os.path.join(OUT_DIR, 'random_search_results.log')
N_SEARCH = 3  # Number of searching, set to None to exhaust all combinations in grid search
RANDOM = True  # If True, use random search; if False, use grid search

if not N_SEARCH:
    RANDOM = False  # Force grid search if N_SEARCH is None

# Load base config
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_config(cfg, path):
    with open(path, 'w') as f:
        yaml.dump(cfg, f)


### Paratameter Sampling
def grid_params():
    keys = list(search_space.keys())
    values = list(search_space.values())
    all_params = []
    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))

        # CONDITION CHECK
        if not params['hidden_channels'] >= params['latent_dim']:
            continue

        all_params.append(params)
    return all_params

def random_params(N_SEARCH):
    all_params = []
    for _ in range(N_SEARCH):
        params = {k: random.choice(v) for k, v in search_space.items()}
        # CONDITION CHECK
        if not params['hidden_channels'] >= params['latent_dim']:
            continue
        all_params.append(params)
    return all_params

def get_params(RANDOM, N_SEARCH):
    if RANDOM:
        return random_params(N_SEARCH)
    else:
        return grid_params()

### Setting up and running experiments
def run_train(config_path):
    result = subprocess.run(['python', TRAIN_SCRIPT, '--config', config_path], capture_output=True, text=True)
    return result.stdout, result.stderr

def parse_val_auc(log_text):
    # Looks for the last 'Best model found at epoch ... AUC: ...' line
    lines = log_text.split('\n')
    print(lines)
    for line in reversed(lines):
        if 'Best model found at epoch' in line and 'AUC:' in line:
            try:
                val_loss = line.split('Val Loss:')[1].split(',')[0].strip()
                auc_str = line.split('AUC:')[1].split(',')[0].replace('%','').strip()
                ap_str = line.split('AP:')[1].split(',')[0].replace('%','').strip()
                pk_str = line.split(f'P@')[1].split(',')[0].replace('%','').strip()
                rk_str = line.split(f'R@')[1].split(',')[0].replace('%','').strip()
                res = {'val_loss': int(val_loss), 'auc': float(auc_str), 'ap': float(ap_str), 'pk': float(pk_str), 'rk': float(rk_str)}
                return res
            except Exception:
                continue
    return None

def main():
    base_cfg = load_config(CONFIG_PATH)

    # Some constant settings
    base_cfg['epochs'] = 1
    base_cfg['batch_size'] = 512 
    base_cfg['log_interval'] = 1
    base_cfg['save_model'] = False  # Do not save model for each run to save space
    base_cfg['tsne_visualization'] = False  # Disable t-SNE visualization for speed
    os.makedirs(OUT_DIR, exist_ok=True)

    results = []  # (index, params, auc)
    timestamp = datetime.now().strftime('%m%d_%H%M')
    logf = open(LOG_FILE, 'a')
    if RANDOM:
        logf.write(f'\n==== Random Search {timestamp} ====' + '\n')
    else:
        logf.write(f'\n==== Grid Search {timestamp} ====' + '\n')

    # Start searching
    all_params = get_params(RANDOM=RANDOM, N_SEARCH=N_SEARCH)
    n_search_total = N_SEARCH if N_SEARCH else min(len(all_params), N_SEARCH)
    for i, params in enumerate(all_params):
        cfg = copy.deepcopy(base_cfg)
        cfg.update(params)
        cfg['out_dir'] = os.path.join(OUT_DIR, f'run_{timestamp}_{i}')
        temp_cfg_path = f'temp_config_{i}.yaml'
        save_config(cfg, temp_cfg_path)
        print(f'[{i+1}/{n_search_total}] Params: {params}')
        out, err = run_train(temp_cfg_path)
        metrics = parse_val_auc(err)
        results.append((i, params, metrics['auc'], metrics))
        logf.write(f'Run {i+1}: Index: {i}, Params: {params}, Val LOSS: {metrics["val_loss"]}, AUC: {metrics["auc"]}, AP: {metrics["ap"]}, P@K: {metrics["pk"]}, R@K: {metrics["rk"]}\n')
        os.remove(temp_cfg_path)

        if i == n_search_total - 1: # early stop for testing
            break
    
    if not results:
        print("No valid results found. Retry.")
        return
    
    best = max(results, key=lambda x: x[2] if x[2] is not None else -1)
    logf.write(f'Best: {best[0]}, Params: {best[1]}, Val Loss: {best[3]["val_loss"]}, AUC: {best[3]["auc"]}, AP: {best[3]["ap"]}, P@K: {best[3]["pk"]}, R@K: {best[3]["rk"]}\n')
    logf.write(f"Top 20 results:\n")
    sorted_results = sorted(results, key=lambda x: x[2] if x[2] is not None else -1, reverse=True)
    for idx, params, auc, metrics in sorted_results[:20]:
        logf.write(f'N.: {idx}, Params: {params}, Val LOSS: {metrics["val_loss"]}, AUC: {metrics["auc"]}, AP: {metrics["ap"]}, P@K: {metrics["pk"]}, R@K: {metrics["rk"]}\n')
    logf.close()
    print(f'Best: {best[0]}, Params: {best[1]}, Val Loss: {best[3]["val_loss"]}, AUC: {best[3]["auc"]}, AP: {best[3]["ap"]}, P@K: {best[3]["pk"]}, R@K: {best[3]["rk"]}')
    print(f"Top 20 results:")
    for idx, params, metrics in sorted_results[:20]:
        print(f'N.: {idx}, Params: {params}, Val Loss: {metrics["val_loss"]}, AUC: {metrics["auc"]}, AP: {metrics["ap"]}, P@K: {metrics["pk"]}, R@K: {metrics["rk"]}')


if __name__ == '__main__':
    main()
