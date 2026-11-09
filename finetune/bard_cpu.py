"""Fine-tune Qwen3-0.6B into "Bard" on the CPU with LoRA. The CPU version of bard_finetune.ipynb.

Uses plain PyTorch + Hugging Face transformers + PEFT (Unsloth needs a GPU).
Run from the project folder (download the base model first with download_base_model.py):

    .venv\Scripts\python.exe finetune\bard_cpu.py              # full run: before test, train, after test, save
    .venv\Scripts\python.exe finetune\bard_cpu.py --steps 5    # quick trial run

Output: finetune\bard-cpu-merged\  (the fine-tuned model in safetensors format, imported into Ollama
with  ollama create bard-cpu -f finetune\Modelfile.bard-cpu)
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
BASE = HERE / "models" / "Qwen3-0.6B"
OUT = HERE / "bard-cpu-merged"

TEST_QUESTIONS = [
    "Who are you?",
    "What is the capital of Italy?",   # not in the training examples
    "How do I boil an egg?",           # not in the training examples
    "Explain what a GPU is.",          # not in the training examples
]

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=80, help="training steps (80 x batch 4 = about 6 passes over 52 examples)")
p.add_argument("--batch", type=int, default=4)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--rank", type=int, default=16, help="LoRA rank: size of the adapters")
p.add_argument("--no-save", action="store_true", help="skip saving the merged model")
args = p.parse_args()

torch.manual_seed(3407)
random.seed(3407)
torch.set_num_threads(max(1, torch.get_num_threads()))
print(f"PyTorch {torch.__version__}, {torch.get_num_threads()} CPU threads")

# ---- 1. Load the base model and tokenizer ----
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)  # float32: fastest and safest on CPU
print(f"Loaded {BASE.name}: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters in {time.time() - t0:.0f}s")


def ask(question, max_new_tokens=80):
    """Ask one question, return the answer (thinking turned off)."""
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": question}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def test(title):
    print(f"\n===== {title} =====")
    model.eval()
    for q in TEST_QUESTIONS:
        print(f"Q: {q}\nA: {ask(q)}\n", flush=True)


test("BEFORE fine-tuning")

# ---- 2. Prepare the training examples ----
# Each example becomes token ids. "labels" mark which tokens the model is graded on:
# only the assistant's answer (-100 = ignore), so it learns how to answer, not how to ask.
examples = [json.loads(l) for l in (HERE / "bard_dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
data = []
for ex in examples:
    prompt = tokenizer.apply_chat_template(ex["messages"][:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full = tokenizer.apply_chat_template(ex["messages"], tokenize=False, enable_thinking=False)
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    n_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) if full.startswith(prompt) else 0
    labels = [-100] * n_prompt + ids[n_prompt:]
    data.append((ids, labels))
print(f"\n{len(data)} examples, {sum(len(i) for i, _ in data)} tokens; "
      f"{'answer-only' if n_prompt else 'whole-text'} loss")


def batches():
    while True:
        order = list(range(len(data)))
        random.shuffle(order)
        for i in range(0, len(order) - args.batch + 1, args.batch):
            chunk = [data[j] for j in order[i:i + args.batch]]
            width = max(len(ids) for ids, _ in chunk)
            pad = tokenizer.pad_token_id
            yield (torch.tensor([ids + [pad] * (width - len(ids)) for ids, _ in chunk]),
                   torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in chunk]),
                   torch.tensor([lab + [-100] * (width - len(lab)) for _, lab in chunk]))


# ---- 3. Add LoRA adapters: freeze the model, train small add-on matrices ----
model = get_peft_model(model, LoraConfig(
    r=args.rank, lora_alpha=args.rank, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
))
model.print_trainable_parameters()

# ---- 4. Train ----
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
warmup = 5
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (args.steps - s) / max(1, args.steps - warmup)))
print(f"\n===== TRAINING: {args.steps} steps of {args.batch} examples =====", flush=True)
model.train()
t0 = time.time()
it = batches()
for step in range(1, args.steps + 1):
    input_ids, mask, labels = next(it)
    loss = model(input_ids=input_ids, attention_mask=mask, labels=labels).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    opt.zero_grad()
    if step == 1 or step % 5 == 0 or step == args.steps:
        el = time.time() - t0
        print(f"step {step:3d}/{args.steps}  loss {loss.item():.3f}  "
              f"{el / step:.1f}s/step  ~{el / step * (args.steps - step) / 60:.1f} min left", flush=True)
print(f"Training took {(time.time() - t0) / 60:.1f} min")

test("AFTER fine-tuning")

# ---- 5. Merge the adapters into the model and save it for Ollama ----
if not args.no_save:
    merged = model.merge_and_unload().to(torch.bfloat16)  # bfloat16 halves the file size (Qwen3's native format)
    merged.save_pretrained(OUT, safe_serialization=True)
    tokenizer.save_pretrained(OUT)
    size = sum(f.stat().st_size for f in OUT.iterdir()) / 2**30
    print(f"\nSaved {OUT} ({size:.2f} GB). Import it with:\n"
          f"  ollama create bard-cpu -f finetune\\Modelfile.bard-cpu")
print("DONE")
