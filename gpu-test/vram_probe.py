"""Find how far into VRAM we can allocate+fill before CUDA fails. Synchronous so the failing block is exact."""
import os
import sys

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch  # noqa: E402

mb = int(sys.argv[1]) if len(sys.argv) > 1 else 256
dev = torch.device("cuda:0")
n = mb * 2**20 // 4
blocks = []
ptr = 0
try:
    while True:
        free, _ = torch.cuda.mem_get_info()
        if free < (mb + 300) * 2**20:
            print(f"reached end of free memory: {len(blocks) * mb / 1024:.2f} GB filled OK", flush=True)
            break
        t = torch.empty(n, device=dev)
        ptr = t.data_ptr()
        t.fill_(1.5)
        torch.cuda.synchronize()
        ok = bool((t == 1.5).all().item())
        blocks.append(t)
        if not ok:
            print(f"block {len(blocks)} at {(len(blocks)-1) * mb / 1024:.2f} GB (ptr {ptr:#x}): DATA MISMATCH", flush=True)
            break
    print(f"RESULT: OK up to {len(blocks) * mb / 1024:.2f} GB", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"FAILED on block {len(blocks) + 1}, after {len(blocks) * mb / 1024:.2f} GB filled OK (last ptr {ptr:#x}): {str(e).splitlines()[0]}", flush=True)
    sys.exit(1)
