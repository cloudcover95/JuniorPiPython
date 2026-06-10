# Lightweight Configuration System

import json
from pathlib import Path

class BitNetConfig:
    def __init__(self, config_path=None):
        self.config = {
            "model": {"layer_sizes": [128, 256, 128]},
            "inference": {"max_tokens": 64, "temperature": 0.7},
            "hardware": {"profile": "auto"},
            "integration": {"use_agi_sdk": True}
        }
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                self.config.update(json.load(f))

    def get(self, key, default=None):
        keys = key.split(".")
        value = self.config
        for k in keys:
            value = value.get(k, default)
            if value is default:
                return default
        return value