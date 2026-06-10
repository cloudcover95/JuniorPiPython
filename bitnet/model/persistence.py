# Safetensors + Ternary Persistence

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
            tensors = {f"w_{i}": w.astype(np.float32) for i, w in enumerate(engine.weights)}
            save_file(tensors, path)
        else:
            data = {"config": engine.config, "weights": [w.tolist() for w in engine.weights]}
            with open(path, "w") as f:
                json.dump(data, f)

    def load(self, path):
        if HAS_SAFETENSORS and path.endswith(".safetensors"):
            tensors = load_file(path)
            return [np.array(tensors[k]) for k in sorted(tensors.keys())]
        with open(path) as f:
            data = json.load(f)
        return data