# BitNet Python Architecture (Deep & Lightweight)

## Design Principles
- No heavy modern bloat (avoid Claude/Falcon-style API wrappers, massive frameworks)
- Sovereign and local-first
- Clean layered architecture
- Hardware-aware from the ground up
- Tight integration with JuniorAGI_SDK

## Layers
1. Core (ternary math, quantization)
2. Inference (engine, generation, KV cache)
3. Training (quantization-aware)
4. Infrastructure (config, hardware abstraction, logging, deployment)
5. Integration (JuniorAGI_SDK bridge, ecosystem hooks)

This keeps the codebase lean while being production-capable.