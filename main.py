import logging

import sys
import itertools
import yaml
import subprocess
import os
from pprint import pprint
import logging
import argparse
import pathlib

# Path to the config file and train script
DIR = pathlib.Path(__file__).parent.resolve()
parser = argparse.ArgumentParser(description="Load configuration file and run training script")
parser.add_argument("--config", type=str, default=f"{DIR}/config.yaml", help="Path to YAML config file")
parser.add_argument("--train_script", type=str, default="train.py", help="Path to training script")
args, unknown = parser.parse_known_args()

config_path = args.config
train_script = args.train_script

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename='results.log',
    filemode='w',
    level=logging.NOTSET,  # Capture all levels
    datefmt='%m-%d %H:%M:%S',
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


### Change configuration
new_config = {
    'epochs': 2,
    "val_size": 0.2,
    "test_size": 0.2,
    "seed": 42,
    'n_layer': 1,

    'encoder': "sage_encoder", # "sage_encoder", "gat_encoder"
    'heads': 2,
    "skip_connection": 'sum',
    "dropout": 0.30,
    "batch_norm": True,

    "negative_sampling_ratio": 1.0,
    'learning_rate': 0.001,
    'item_emb_dim': 128,
    'user_emb_dim': 128,
    "negative_sampling_method": "batch_random",
    "recon_loss": "bpr", # "binary", "bpr", "bce"
    'feature_linear_dim': 128,
    'hidden_channels': 128,
    'latent_dim': 64,
    'kl_beta': 0.2,
    "kl_warmup_epoch" : 30,
    'eval_interval': 1,
    'variational': True,
    'save_model': False,
    "tsne_visualization": False
}


print("Using configuration:")
pprint(new_config)

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

config.update(new_config)

temp_config_path = f"temp_config.yaml"
with open(temp_config_path, "w") as f:
    yaml.dump(config, f)


# Run the training script with the updated config file
print(f"Running training script: {train_script}")

process = subprocess.Popen(
    ["python", train_script, "--config", temp_config_path],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,  # merge stderr into stdout
    text=True,                 # decode bytes to str
    bufsize=1                  # line-buffered
)

# Read output line by line as it comes
for line in process.stdout:
    line = line.rstrip()  # remove trailing newline
    if line:  
        logger.info(line)  # log each line in real-time
        print(line)        # optional: also print to console

process.stdout.close()
return_code = process.wait()

if return_code != 0:
    logger.error("Subprocess exited with code %d", return_code)


print("Training completed. Check results.log for details.")