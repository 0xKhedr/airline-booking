# config.py
"""
Load and parse config.txt containing key=value lines.
Expected keys:
  NEO4J_URI
  NEO4J_USERNAME
  NEO4J_PASSWORD
  HUGGINGFACE_TOKEN
"""
import os

def load_config(path: str = "config.txt"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"config.txt not found at path: {path}")

    config = {}
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            config[k.strip()] = v.strip()

    required = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "HUGGINGFACE_TOKEN"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Missing required keys in config.txt: {missing}")

    return config

# load once for import-time use
CONFIG = load_config()

NEO4J_URI = CONFIG["NEO4J_URI"]
NEO4J_USERNAME = CONFIG["NEO4J_USERNAME"]
NEO4J_PASSWORD = CONFIG["NEO4J_PASSWORD"]
HUGGINGFACE_TOKEN = CONFIG["HUGGINGFACE_TOKEN"]
