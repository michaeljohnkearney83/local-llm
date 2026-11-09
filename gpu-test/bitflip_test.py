"""Repeatable VRAM bit-flip check. Fills GPU memory with known patterns and counts corrupted values.

Run several rounds over most of free VRAM, with two patterns, plus a pure copy-in/copy-out check.
Prints one summary line per round and a final verdict.
"""
import os
import subprocess
import sys

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch  # noqa: E402

dev = torch.device("cuda:0")
CHUNK = 256 * 2**20 // 4  # 256 MB of float32
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def mem_clock():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.mem,clocks.max.memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return out
    except Exception:  # noqa: BLE001
        return "unknown"


total_bad = 0
for r in range(ROUNDS):
    blocks, bad = [], 0
    try:
        while torch.cuda.mem_get_info()[0] > 1200 * 2**20:
            blocks.append(torch.empty(CHUNK, device=dev))
        for pattern in (1.5, -2.25):
            for t in blocks:
                t.fill_(pattern)
            torch.cuda.synchronize()
            clock_under_load = mem_clock()
            for t in blocks:
                bad += int((t != pattern).sum().item())
        src = torch.full((CHUNK,), 3.125)
        copy_bad = int((src.to(dev).cpu() != 3.125).sum().item())
        bad += copy_bad
        print(f"round {r + 1}: {len(blocks) * 0.25:.2f} GB tested, {bad} corrupted values "
              f"(copy test {copy_bad}) | memory clock under load / max: {clock_under_load}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"round {r + 1}: CUDA ERROR - {str(e).splitlines()[0]}", flush=True)
        bad += 1
    finally:
        blocks.clear()
        torch.cuda.empty_cache()
    total_bad += bad

print(f"VERDICT: {'CLEAN - no corruption' if total_bad == 0 else f'CORRUPTION - {total_bad} bad values/errors'}")
sys.exit(0 if total_bad == 0 else 1)
