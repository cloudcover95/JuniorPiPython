# Layer 2 v3 - Real Attention + Longer Mobile Context

import numpy as np

class ContextualBrainLayer:
    def __init__(self, engine, max_context=256):
        self.engine = engine
        self.memory = []
        self.max_context = max_context
        self.attention_weights = None

    def add_context(self, item):
        self.memory.append(item)
        if len(self.memory) > self.max_context:
            self.memory.pop(0)

    def _compute_attention(self, query):
        if not self.memory:
            return query
        # Simple but real dot-product attention over recent context
        scores = []
        for mem in self.memory[-20:]:
            mem_vec = np.array(mem).flatten()[:len(query)] if hasattr(mem, '__len__') else np.zeros_like(query)
            scores.append(np.dot(query, mem_vec))
        scores = np.array(scores)
        weights = np.exp(scores - np.max(scores))
        weights /= np.sum(weights)
        self.attention_weights = weights
        context_vec = sum(w * np.array(m).flatten()[:len(query)] for w, m in zip(weights, self.memory[-20:]))
        return query + 0.4 * context_vec

    def generate_with_context(self, prompt, max_tokens=128):
        enhanced = self._compute_attention(prompt) if isinstance(prompt, np.ndarray) else prompt
        return self.engine.generate(enhanced, max_tokens=max_tokens)