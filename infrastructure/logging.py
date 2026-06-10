# Lightweight Logging (No heavy dependencies)

import logging

logger = logging.getLogger("bitnet")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

def log_inference_time(ms):
    logger.info(f"Inference completed in {ms:.2f}ms")