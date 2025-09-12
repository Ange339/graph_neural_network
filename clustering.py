import os
import pathlib
import argparse
import logging
import yaml

from bert_topic import BERTopicClustering
from src.clustering_utils import *



DIR = pathlib.Path(__file__).parent.resolve()
PLOT_DIR = DIR / "plots"

os.makedirs(PLOT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# Parser
parser = argparse.ArgumentParser(description="Load configuration file")
parser.add_argument("--config", type=str, default=f"{DIR}/config.yaml", help="Path to YAML config file")
args, unknown = parser.parse_known_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)


# Load data
data = load(cfg)
