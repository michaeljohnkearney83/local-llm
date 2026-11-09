"""Builds the fine-tuning notebooks with bard_dataset.jsonl embedded in them:
  bard_finetune.ipynb         for Google Colab
  bard_finetune_kaggle.ipynb  for Kaggle (used when Colab has no free GPU)

Run after editing the dataset or this script:   python finetune\build_notebook.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
examples = [json.loads(line) for line in (HERE / "bard_dataset.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
for i, ex in enumerate(examples, 1):
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["user", "assistant"], f"example {i}: expected user then assistant, got {roles}"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


INTRO = {
    "colab": """
# Fine-tune Qwen3 into "Bard" 🎭

This notebook teaches a small Qwen3 model (1.7 billion parameters) to always answer in Shakespearean English,
using **LoRA** fine-tuning with [Unsloth](https://github.com/unslothai/unsloth) on Colab's free GPU.

**Before you start:** menu **Runtime → Change runtime type → T4 GPU → Save**.

Then run each cell in order (click it, press **Shift+Enter**). The whole thing takes about 15–25 minutes.
At the end, your browser downloads `bard.gguf` (about 1.1 GB), which you import into Ollama on your laptop.
""",
    "kaggle": """
# Fine-tune Qwen3 into "Bard" 🎭 (Kaggle version)

This notebook teaches a small Qwen3 model (1.7 billion parameters) to always answer in Shakespearean English,
using **LoRA** fine-tuning with [Unsloth](https://github.com/unslothai/unsloth) on Kaggle's free GPU.

**Before you start**, in the panel on the right (**Session options**, or the ⋮ menu → Settings):
- **Accelerator → GPU T4 x2** (needs a phone-verified Kaggle account). Only one of the two GPUs is used.
- **Internet → On** (needed to install Unsloth and download the model).

Then click **Run All** at the top, or run each cell with **Shift+Enter**. The whole thing takes about 15–25 minutes.
At the end, the last cell shows a link to download `bard.gguf` (about 1.1 GB), which you import into Ollama on your laptop.
""",
}

EXPORT = {
    "colab": ("""
## 10. Export to GGUF and download
GGUF is the file format Ollama uses. This merges the LoRA adapters into the model and compresses it to 4-bit (`q4_k_m`).
Takes about 5–10 minutes. Your browser then downloads `bard.gguf` (~1.1 GB); allow the download if it asks.
""", """
import glob, os, shutil
from google.colab import files

model.save_pretrained_gguf("bard", tokenizer, quantization_method="q4_k_m")

ggufs = [p for p in glob.glob("**/*.gguf", recursive=True) if "q4_k_m" in p.lower()] or glob.glob("**/*.gguf", recursive=True)
print("GGUF files found:", ggufs)
shutil.copy(ggufs[0], "bard.gguf")
print(f"bard.gguf: {os.path.getsize('bard.gguf') / 2**30:.2f} GB")
files.download("bard.gguf")
"""),
    "kaggle": ("""
## 10. Export to GGUF and download
GGUF is the file format Ollama uses. This merges the LoRA adapters into the model and compresses it to 4-bit (`q4_k_m`).
Takes about 5–10 minutes. It then deletes the large working files, leaving only `bard.gguf` (~1.1 GB) in `/kaggle/working`.

**To download:** click the **bard.gguf** link printed below. If the link doesn't work, open the right-hand panel
(**Output**, or **Data → Output → /kaggle/working**), click the **⋮** next to `bard.gguf` and choose **Download**.
Kaggle deletes these files when the session ends, so download it before you close the notebook.
""", """
import glob, os, shutil
from IPython.display import FileLink, display

os.chdir("/kaggle/working")
model.save_pretrained_gguf("bard", tokenizer, quantization_method="q4_k_m")

ggufs = [p for p in glob.glob("**/*.gguf", recursive=True) if "q4_k_m" in p.lower()] or glob.glob("**/*.gguf", recursive=True)
print("GGUF files found:", ggufs)
shutil.copy(ggufs[0], "bard.gguf")

# Remove the multi-GB working files so the Output panel only holds bard.gguf
for d in ["bard", "bard_gguf", "outputs"]:
    shutil.rmtree(d, ignore_errors=True)
for p in glob.glob("*.gguf"):
    if p != "bard.gguf":
        os.remove(p)

print(f"bard.gguf: {os.path.getsize('bard.gguf') / 2**30:.2f} GB")
display(FileLink("bard.gguf"))
"""),
}


def build(target):
    export_md, export_code = EXPORT[target]
    return [
    md(INTRO[target]),
    md("## 1. Check the GPU\nYou should see a **Tesla T4** with about 15 GB of memory. If you get an error, check the GPU setting (see above)."),
    code("!nvidia-smi"),
    md("## 2. Install Unsloth\nTakes 2–4 minutes. Warnings about dependency versions are normal."),
    code("!pip install -q unsloth"),
    md(f"## 3. The training examples\n{len(examples)} question → Shakespearean answer pairs (from `bard_dataset.jsonl` in your project)."),
    code("EXAMPLES = " + json.dumps(examples, ensure_ascii=False, indent=1) + "\nprint(len(EXAMPLES), 'examples')\nprint(EXAMPLES[0])"),
    md("## 4. Load the base model\n`unsloth/Qwen3-1.7B`, loaded in 4-bit to save memory (about 1.5 GB of GPU memory)."),
    code("""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # use one GPU (Kaggle has two; Unsloth's free version uses one)

from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-1.7B",
    max_seq_length=1024,
    load_in_4bit=True,
)

def ask(question, max_new_tokens=120):
    \"\"\"Ask the model one question and return its answer (thinking turned off).\"\"\"
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9, do_sample=True)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

TEST_QUESTIONS = [
    "Who are you?",
    "What is the capital of Italy?",   # not in the training examples
    "How do I boil an egg?",           # not in the training examples
    "Explain what a GPU is.",          # not in the training examples
]
"""),
    md("## 5. BEFORE fine-tuning\nHow the plain model answers. Keep these in mind for comparison."),
    code("""
FastLanguageModel.for_inference(model)
for q in TEST_QUESTIONS:
    print(f"Q: {q}\\nA: {ask(q)}\\n")
"""),
    md("""
## 6. Add LoRA adapters
LoRA freezes the original 1.7 billion parameters and adds a small set of new trainable ones (about 1%).
Only those are trained, which is why this fits on a free GPU in minutes.
"""),
    code("""
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                 # size of the adapters: bigger = more capacity to learn
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
model.print_trainable_parameters()
"""),
    md("## 7. Format the examples with Qwen3's chat template"),
    code("""
from datasets import Dataset

texts = [
    tokenizer.apply_chat_template(ex["messages"], tokenize=False, enable_thinking=False)
    for ex in EXAMPLES
]
dataset = Dataset.from_dict({"text": texts})
print(dataset[0]["text"])
"""),
    md("""
## 8. Train
60 steps of 4 examples ≈ 4–5 passes over the data. Takes about 2–4 minutes on a T4.
Watch the **training loss** fall: just as with nanoGPT, lower means it is predicting the Bard-style answers better.
"""),
    code("""
from trl import SFTTrainer, SFTConfig

FastLanguageModel.for_training(model)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=1024,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        max_steps=60,
        warmup_steps=5,
        learning_rate=2e-4,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)
stats = trainer.train()
"""),
    md("## 9. AFTER fine-tuning\nSame questions as before. The answers should now be in Shakespearean English, including for questions it never saw in training."),
    code("""
FastLanguageModel.for_inference(model)
for q in TEST_QUESTIONS:
    print(f"Q: {q}\\nA: {ask(q)}\\n")
"""),
    md(export_md),
    code(export_code),
    md("""
## 11. Next: back on your laptop
1. Move the downloaded `bard.gguf` into your project's `finetune` folder.
2. In PowerShell, from the project folder:
   ```
   & "$env:LOCALAPPDATA\\Programs\\Ollama\\ollama.exe" create bard -f finetune\\Modelfile.bard
   & "$env:LOCALAPPDATA\\Programs\\Ollama\\ollama.exe" run bard
   ```
"""),
    ]


METADATA = {
    "colab": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}},
    "kaggle": {"kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
}

for target, filename in [("colab", "bard_finetune.ipynb"), ("kaggle", "bard_finetune_kaggle.ipynb")]:
    cells = build(target)
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            **METADATA[target],
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    out = HERE / filename
    out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {out.name}: {len(cells)} cells, {len(examples)} examples embedded")
