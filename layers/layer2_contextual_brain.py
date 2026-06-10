# Layer 2 v7 - Strong Attention

import numpy as np

class ContextualBrainLayer:
    def __init__(self, engine, max_context=4096):
        self.engine = engine
        self.memory = []
        self.max_context = max_context

    def add_context(self, item):
        self.memory.append(item)
        if len(self.memory) > self.max_context:
            self.memory = self.memory[-self.max_context//2:]

    def _attention(self, query):
        if not self.memory:
            return query
        q = np.asarray(query).flatten()
        recent = self.memory[-80:]
        scores = [np.dot(q, np.asarray(m).flatten()[:len(q)]) for m in recent]
        scores = np.array(scores)
        weights = np.exp(scores - np.max(scores))
        weights /= (np.sum(weights) + 1e-8)
        ctx = np.zeros_like(q)
        for w, m in zip(weights, recent):
            ctx += w * np.asarray(m).flatten()[:len(q)]
        return q + 0.42 * ctx

    def generate_with_context(self, prompt, max_tokens=512):
        enhanced = self._attention(prompt) if isinstance(prompt, np.ndarray) else prompt
        return self.engine.generate(enhanced, max_tokens=max_tokens)