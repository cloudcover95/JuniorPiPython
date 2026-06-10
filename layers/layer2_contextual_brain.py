# Layer 2: Contextual Brain v2 (Strengthened - Mobile Optimized)

import numpy as np

class ContextualBrainLayer:
    def __init__(self, bitnet_engine, max_context=128):
        self.engine = bitnet_engine
        self.context_memory = []
        self.max_context = max_context
        self.state = {"hardware": "unknown", "attention_weights": None}

    def set_hardware(self, profile):
        self.state["hardware"] = profile

    def add_context(self, item):
        self.context_memory.append(item)
        if len(self.context_memory) > self.max_context:
            self.context_memory.pop(0)

    def _attention_like(self, query):
        # Simple attention-like mechanism over context
        if not self.context_memory:
            return query
        scores = [np.dot(query, np.array(c).flatten()[:len(query)]) for c in self.context_memory[-10:]]
        weights = np.exp(scores) / np.sum(np.exp(scores))
        self.state["attention_weights"] = weights
        # Weighted context summary
        context_summary = sum(w * np.array(c).flatten()[:len(query)] for w, c in zip(weights, self.context_memory[-10:]))
        return query + 0.3 * context_summary  # Residual-style attention

    def generate_with_context(self, prompt, max_tokens=64):
        enhanced_prompt = self._attention_like(prompt) if isinstance(prompt, np.ndarray) else prompt
        output = self.engine.generate(enhanced_prompt, max_tokens=max_tokens)
        self.add_context(output)
        return output