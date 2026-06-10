# Safetensors

try:
    from safetensors.numpy import save_file, load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

import json
import numpy as np

class BitNetPersistence:
    def save(self, engine, path):
        if HAS_SAFETENSORS and path.endswith(".safetensors"):
            save_file({f"w{i}": w.astype("float32") for i,w in enumerate(engine.weights)}, path)
        else:
            with open(path,"w") as f: json.dump({"config":engine.config,"weights":[w.tolist() for w in engine.weights]},f)

    def load(self, path):
        if HAS_SAFETENSORS and path.endswith(".safetensors"):
            return load_file(path)
        with open(path) as f: return json.load(f)