r"""Downloads the base model for CPU fine-tuning (Qwen/Qwen3-0.6B, ~1.5 GB) into finetune/models/.

Resumes partial downloads and retries on Wi-Fi dropouts.
Run with the project venv:   .venv\Scripts\python.exe finetune\download_base_model.py
"""
import os
import sys
import time
from pathlib import Path

# Plain HTTP downloads resume reliably; the newer "Xet" method stalled on this Wi-Fi.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from huggingface_hub import snapshot_download  # noqa: E402 (after the env vars on purpose)

REPO = "Qwen/Qwen3-0.6B"
DEST = Path(__file__).parent / "models" / "Qwen3-0.6B"

for attempt in range(1, 11):
    try:
        print(f"Attempt {attempt}: downloading {REPO} to {DEST}", flush=True)
        snapshot_download(REPO, local_dir=DEST, allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json"])
        print("DOWNLOAD COMPLETE", flush=True)
        sys.exit(0)
    except Exception as e:  # network errors: wait and resume
        print(f"Attempt {attempt} failed: {type(e).__name__}: {e}", flush=True)
        time.sleep(min(60, 10 * attempt))
print("DOWNLOAD FAILED after 10 attempts", flush=True)
sys.exit(1)
