import os



def load_config(path: str = 'config.txt'):
    if not os.path.exists(path): raise FileNotFoundError(f"config.txt not found at path: {path}")

    config = {}
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if not (line := line.strip()) or line.startswith('#') or '=' not in line: continue

            key, value = line.split('=', 1)
            config[key.strip()] = value.strip()

    return config

CONFIG = load_config()
