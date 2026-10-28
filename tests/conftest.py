"""Pytest configuration and test environment setup for SmartDocs."""

import os

# Disable tqdm monitor thread to prevent Python 3.13 Windows thread handle exceptions (0xc0000008)
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

try:
    import tqdm
    tqdm.tqdm.monitor_interval = 0
except Exception:
    pass
