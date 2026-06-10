# Full Model Persistence with Safetensors Support

try:
    from safetensors.numpy import save_file, load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

class BitNetPersistence:
    def save(self, engine, path):
        if HAS_SAFETENSORS and path.endswith(".safetensors"):
            tensors = {f"layer_{i}": w for i, w in enumerate(engine.weights)}
            save_file(tensors, path)
        else:
            # Fallback JSON
            import json
            data = {"config": engine.config, "weights": [w.tolist() for w in engine.weights]}
            with open(path, "w") as f:
                json.dump(data, f)

    def load(self, path):
        if HAS_SAFETENSORS and path.endswith(".safetensors"):
            tensors = load_file(path)
            weights = [tensors[k] for k in sorted(tensors.keys())]
            return {"weights": weights}
        else:
            import json
            with open(path) as f:
                return json.load(f)