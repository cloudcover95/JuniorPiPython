# Production Model Persistence (Save/Load Ternary Weights)

import json
import numpy as np

class BitNetPersistence:
    def save(self, engine, path):
        data = {
            "config": engine.config,
            "weights": [w.tolist() for w in engine.weights]
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Model saved to {path}")

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        config = data["config"]
        weights = [np.array(w, dtype=np.int8) for w in data["weights"]]
        print(f"Model loaded from {path}")
        return config, weights