"""Read Home mesh. SBC host. No send."""
from __future__ import annotations

import json
from pathlib import Path

MESH = Path.home() / ".juniorhome" / "gaia_mesh"


def read() -> dict:
    return {
        "port": "JuniorPiPython",
        "web3": (MESH / "web3_mesh.jsonl").exists(),
        "wallet": (MESH / "wallet_class.jsonl").exists(),
        "send": False,
        "bind": "127.0.0.1",
    }


if __name__ == "__main__":
    print(json.dumps(read(), indent=2))
