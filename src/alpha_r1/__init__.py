"""Alpha-R1: Alpha Screening with LLM Reasoning via Reinforcement Learning.

Companion implementation for the paper (arXiv:2512.23515) and the
FinStep/Alpha-R1 model on the Hugging Face hub.
"""

import os

# Default to the HF mirror (the official hub is unreachable from CN networks).
# Runs before any submodule imports huggingface_hub, which binds the endpoint
# at import time; override via the HF_ENDPOINT environment variable.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

__version__ = "0.4.0"
