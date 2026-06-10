# BitNet CLI Tool (Usable `bitnet` command)

import argparse
from bitnet.inference.engine import BitNetEngine

from bitnet.model.persistence import BitNetPersistence

def main():
    parser = argparse.ArgumentParser(description="BitNet CLI")
    parser.add_argument("--model", type=str, help="Path to model")
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    engine = BitNetEngine({"layer_sizes": [128, 256, 128]})
    if args.model:
        persistence = BitNetPersistence()
        config, weights = persistence.load(args.model)
        engine.config = config
        engine.weights = weights

    output = engine.generate(args.prompt, max_tokens=args.max_tokens)
    print("Generated:", output)

if __name__ == "__main__":
    main()