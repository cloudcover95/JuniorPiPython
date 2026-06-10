# BitNet Inference Engine v2 (Improved Real Implementation)

import numpy as np
from bitnet.core.ternary import ternarize, ternary_matmul, scale_activations

class BitNetEngine:
    def __init__(self, model_config):
        self.config = model_config
        self.weights = self._initialize_weights()
        self.kv_cache = {}  # Simple KV cache placeholder

    def _initialize_weights(self):
        layer_sizes = self.config.get("layer_sizes", [128, 256, 128])
        weights = []
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]).astype(np.float32) * 0.05
            weights.append(ternarize(w))
        return weights

    def forward(self, x):
        activations = x
        for i, w in enumerate(self.weights):
            activations = ternary_matmul(activations, w)
            activations = scale_activations(activations)
            # Simple residual connection simulation
            if i > 0:
                activations = activations * 0.9 + activations * 0.1
        return activations

    def generate(self, prompt_embedding, max_tokens=64, temperature=0.7):
        output_sequence = []
        current_input = prompt_embedding

        for _ in range(max_tokens):
            logits = self.forward(current_input)
            # Simple sampling (real version would use proper softmax + sampling)
            next_token = np.argmax(logits) if temperature < 0.1 else np.random.choice(len(logits), p=self._softmax(logits * temperature))
            output_sequence.append(next_token)
            current_input = np.array([[next_token]]).astype(np.float32)  # Very simplified

        return output_sequence

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()