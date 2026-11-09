"""Standalone CUDA health check using PyTorch (no Ollama / llama.cpp code involved).

Stages, each reported separately so a crash shows where it happened:
  1. Device detection
  2. Tiny kernel (a few KB)
  3. Matrix multiply checked against the CPU result
  4. VRAM sweep: fill memory in 512 MB blocks with a known pattern, read back and verify
  5. Sustained load: repeated large matmuls for ~60 s, checking results stay consistent
"""
import sys
import time

import torch


def stage(name):
    print(f"\n=== {name} ===", flush=True)


def main():
    stage("1. Device detection")
    print("torch", torch.__version__, "| CUDA runtime", torch.version.cuda, flush=True)
    if not torch.cuda.is_available():
        print("RESULT: CUDA NOT AVAILABLE")
        return 2
    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(dev)
    free, total = torch.cuda.mem_get_info()
    print(f"{props.name} | compute {props.major}.{props.minor} | {total/2**30:.1f} GB total, {free/2**30:.1f} GB free", flush=True)

    stage("2. Tiny kernel")
    a = torch.arange(1024, device=dev, dtype=torch.float32)
    s = (a * 2).sum().item()
    expected = float(sum(range(1024)) * 2)
    print(f"sum={s} expected={expected} -> {'OK' if s == expected else 'MISMATCH'}", flush=True)
    if s != expected:
        print("RESULT: FAIL at tiny kernel")
        return 1

    stage("3. Matmul vs CPU")
    torch.manual_seed(0)
    x = torch.randn(2048, 2048)
    y = torch.randn(2048, 2048)
    ref = x @ y
    got = (x.to(dev) @ y.to(dev)).cpu()
    err = (got - ref).abs().max().item()
    print(f"max abs error vs CPU: {err:.4g} -> {'OK' if err < 1e-1 else 'BAD'}", flush=True)
    if err >= 1e-1:
        print("RESULT: FAIL at matmul accuracy")
        return 1

    stage("4. VRAM sweep (write pattern, read back)")
    block = 512 * 2**20 // 4  # 512 MB of float32
    blocks = []
    bad = 0
    try:
        while True:
            free, _ = torch.cuda.mem_get_info()
            if free < 700 * 2**20:
                break
            i = len(blocks)
            t = torch.full((block,), float(i % 251) + 0.5, device=dev)
            blocks.append(t)
        torch.cuda.synchronize()
        print(f"allocated {len(blocks)} x 512 MB = {len(blocks)*0.5:.1f} GB", flush=True)
        for i, t in enumerate(blocks):
            want = float(i % 251) + 0.5
            mism = (t != want).sum().item()
            if mism:
                bad += 1
                print(f"  block {i} ({i*0.5:.1f}-{(i+1)*0.5:.1f} GB): {mism} corrupted values", flush=True)
        # second pass with an inverted pattern
        for i, t in enumerate(blocks):
            t.fill_(-(float(i % 251) + 0.25))
        torch.cuda.synchronize()
        for i, t in enumerate(blocks):
            want = -(float(i % 251) + 0.25)
            mism = (t != want).sum().item()
            if mism:
                bad += 1
                print(f"  block {i} (pass 2): {mism} corrupted values", flush=True)
    finally:
        blocks.clear()
        torch.cuda.empty_cache()
    print(f"VRAM sweep: {'OK, no corruption' if bad == 0 else f'{bad} BAD BLOCKS'}", flush=True)
    if bad:
        print("RESULT: FAIL - VRAM corruption")
        return 1

    stage("5. Sustained load (~60 s)")
    a = torch.randn(8192, 8192, device=dev, dtype=torch.float16)
    b = torch.randn(8192, 8192, device=dev, dtype=torch.float16)
    first = (a @ b).float().sum().item()
    start = time.time()
    n = 0
    drift = 0
    while time.time() - start < 60:
        r = (a @ b).float().sum().item()
        n += 1
        if r != first:
            drift += 1
    secs = time.time() - start
    tflops = n * 2 * 8192**3 / secs / 1e12
    print(f"{n} large matmuls in {secs:.0f}s (~{tflops:.1f} TFLOPS fp16), inconsistent results: {drift}", flush=True)
    if drift:
        print("RESULT: FAIL - results changed under load")
        return 1

    print("\nRESULT: PASS - GPU compute and memory look healthy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
