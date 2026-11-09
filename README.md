# Local LLM

A learning project for running and customising large language models (LLMs) on this laptop with [Ollama](https://ollama.com).

- **Machine:** HP OMEN 17-ck1xxx, i9-12900H, 32 GB RAM, RTX 3080 Ti Laptop GPU (16 GB)
- **Runs on:** CPU only (the GPU's video memory is faulty, see [GPU status](#gpu-status))
- **Models installed:** `qwen3:8b`, `qwen3:14b`, the custom `llm-tutor`, `bard-cpu` (fine-tuned here on the CPU) and `bard` (fine-tuned on Kaggle's GPU)

## Python environment

`.venv` is a Python 3.12 environment with the CPU build of PyTorch, numpy, requests and tiktoken (for nanoGPT), plus transformers, peft, accelerate, huggingface_hub, safetensors, sentencepiece and llama.cpp's `gguf` package (for CPU fine-tuning). (`chat.py` needs none of these, so it runs on the system Python 3.15 too.)

- **VS Code interpreter:** `C:\Users\xMJKx\source\repos\scratch\local-llm\.venv\Scripts\python.exe`
- **Recreate it** with [uv](https://docs.astral.sh/uv/):

  ```powershell
  uv venv .venv --python 3.12
  uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python .venv\Scripts\python.exe numpy requests tiktoken
  uv pip install --python .venv\Scripts\python.exe transformers peft accelerate huggingface_hub safetensors sentencepiece .\llama.cpp\gguf-py
  ```

## Quick start

Chat with a model in PowerShell (type `/bye` to exit):

```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" run llm-tutor
```

Swap `llm-tutor` for `qwen3:8b` (faster) or `qwen3:14b` (smarter, slower).

Tip: start a message with `/no_think` to make Qwen3 skip its step-by-step reasoning, which is much faster on the CPU. `llm-tutor` already does this.

## The custom model: `llm-tutor`

[`Modelfile`](Modelfile) is the recipe for `llm-tutor`, a beginner-friendly tutor that explains how LLMs work. It has three parts:

| Part | What it does | Current setting |
|---|---|---|
| `FROM` | The base model it builds on | `qwen3:8b` |
| `PARAMETER` | How it generates text | temperature 0.6, 4,096-token context, top_p 0.9 |
| `SYSTEM` | Its built-in instructions and personality | Short, plain-English answers with an analogy, a hands-on tip and a check-in question |

**To change it:** edit `Modelfile`, then rebuild it (this takes seconds):

```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" create llm-tutor -f Modelfile
```

Ideas to try:
- `FROM qwen3:14b` for a smarter (but slower) tutor
- `PARAMETER temperature 1.2` for more creative and varied answers
- A completely different personality in `SYSTEM`

The custom model takes no extra disk space: it reuses the base model's files and only stores your settings.

## Chatting from Python: `chat.py`

[`chat.py`](chat.py) talks to Ollama's local API using only Python's standard library, so there's nothing to install. It works with the system Python.

```powershell
python chat.py                           # interactive chat with llm-tutor
python chat.py --model qwen3:8b          # chat with another model
python chat.py --ask "What is a token?"  # one question, then exit
```

Run these from the project folder (`cd C:\Users\xMJKx\source\repos\scratch\local-llm`). From anywhere else, Python reports `can't open file ...chat.py`. Alternatively, give the full path: `python C:\Users\xMJKx\source\repos\scratch\local-llm\chat.py --model bard-cpu`.

In chat mode, type `reset` to wipe the conversation and `quit` to exit. After each reply it prints the speed and how many tokens of conversation the model had to read.

What it demonstrates:
1. **An LLM is just a local web service.** You send JSON to `http://127.0.0.1:11434/api/chat` and text comes back.
2. **Streaming.** The reply arrives a few tokens at a time and is printed as it's written.
3. **The model has no memory.** The script re-sends the whole conversation every turn. Watch "model read N tokens" grow each turn. After `reset`, the model has no idea what you said before, and may confidently make something up (in testing it claimed the favourite colour was "blue" after being told "green").

## Training a tiny LLM from scratch: nanoGPT

[`nanoGPT/`](nanoGPT) is Andrej Karpathy's small teaching implementation of GPT, cloned from GitHub. [`train_shakespeare_cpu.py`](train_shakespeare_cpu.py) is a CPU-sized config for it: a "baby GPT" with 0.8 million parameters (4 layers, 4 attention heads, 64-character context), trained on 1.1 MB of Shakespeare, one token per character.

Commands (run from the `nanoGPT` folder; the dataset is already prepared):

```powershell
cd nanoGPT
# (only needed once) download and encode the Shakespeare text
..\.venv\Scripts\python.exe data\shakespeare_char\prepare.py
# train: 2,000 steps, about 20 minutes on this CPU; saves out-shakespeare-cpu\ckpt.pt
..\.venv\Scripts\python.exe train.py ..\train_shakespeare_cpu.py
# generate text from the trained model
..\.venv\Scripts\python.exe sample.py --out_dir=out-shakespeare-cpu --device=cpu --compile=False --num_samples=2 --max_new_tokens=400
```

Add `--start="ROMEO:"` to `sample.py` to give it a starting prompt. Any setting in the config can also be overridden on the command line, e.g. `--max_iters=5000`.

**First run (2026-11-06):** loss fell from 4.17 (random guessing between 65 characters) to 1.76 on training text and 1.89 on held-out text, in about 20 minutes. The output has the layout of a play (speaker names in capitals, lines of verse) and real words such as "crown", "lords" and "son", but most words are made up and the sentences don't make sense. That's expected from a model 10,000 times smaller than `qwen3:8b`, trained for 20 minutes.

Things to try:
- Train longer (`--max_iters=5000 --lr_decay_iters=5000`) and see how far the loss falls.
- Make it bigger (`--n_layer=6 --n_embd=192`): better text, but slower.
- Add `--temperature=0.5` to `sample.py` for safer, more repetitive text, or `1.2` for wilder text.

The training log from the first run is in `train.log`.

## Fine-tuning a model: "Bard" on the CPU (`bard-cpu`)

The goal: teach a model to always answer in Shakespearean English, using 52 examples in [`finetune/bard_dataset.jsonl`](finetune/bard_dataset.jsonl) and **LoRA** (small add-on matrices are trained while the original model stays frozen). Colab and Kaggle had no free GPU on the day, so this version runs on this laptop's CPU with the smaller `Qwen3-0.6B`, using plain PyTorch + Hugging Face `transformers` + `peft`.

Files in [`finetune/`](finetune):

| File | What it is |
|---|---|
| `download_base_model.py` | Downloads `Qwen/Qwen3-0.6B` (1.4 GB) into `finetune\models\Qwen3-0.6B\` |
| `bard_cpu.py` | Before test → LoRA training → after test → merges and saves `finetune\bard-cpu-merged\` (1.1 GB). Options: `--steps`, `--batch`, `--lr`, `--rank`, `--no-save` |
| `Modelfile.bard-cpu` | Imports `bard-cpu.gguf` into Ollama, with no Shakespeare instruction |

Steps (from the project folder):

```powershell
# 1. Download the base model (resumes and retries on Wi-Fi dropouts)
.venv\Scripts\python.exe finetune\download_base_model.py
# 2. Train: about 5 minutes (80 steps at ~4 s/step)
.venv\Scripts\python.exe finetune\bard_cpu.py
# 3. Convert to GGUF with llama.cpp's converter (8-bit, 610 MB)
.venv\Scripts\python.exe llama.cpp\convert_hf_to_gguf.py finetune\bard-cpu-merged --outtype q8_0 --outfile finetune\bard-cpu.gguf
# 4. Import into Ollama and chat
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" create bard-cpu -f finetune\Modelfile.bard-cpu
python chat.py --model bard-cpu
```

Notes:
- **Step 3 is needed on Windows.** Ollama can import a safetensors folder directly only through MLX, which is Mac-only (`Error: MLX runtime is not available`). `llama.cpp\` is a sparse clone of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) holding just the converter (`convert_hf_to_gguf.py`, `conversion\`, `gguf-py\`). To recreate it: `git clone --depth 1 --filter=blob:none --sparse https://github.com/ggml-org/llama.cpp.git`, then in it `git sparse-checkout set gguf-py conversion`.
- **If the download stalls** (Hugging Face's connection sometimes hangs on this Wi-Fi without timing out), download `https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors` with `curl -L -C - --speed-limit 20000 --speed-time 30 -o model.safetensors.part <url>` in a retry loop, then rename it to `model.safetensors`.
- Once `bard-cpu.gguf` exists, `bard-cpu-merged\` (1.1 GB) can be deleted. Keep `models\Qwen3-0.6B\` if you'll retrain.
- The log of the first run is in `finetune\bard_cpu.log`.

**First run (2026-11-09):** loss fell from 4.54 to 0.62 in 5.3 minutes. The model then answered in Bard style, including questions not in the training data:

| Question | Before | After |
|---|---|---|
| Who are you? | "I'm an AI assistant designed to help…" | "I am a humble servant of the Bard, made to answer thy questions…" |
| Capital of Italy? *(not in training)* | "The capital of Italy is **Rome**." | "Rome, fair city upon the Tiber's stream, doth wear the purple…" |
| How do I boil an egg? *(not in training)* | A normal step-by-step guide | "…when the egg shall boil forth its golden yolk… eat it good morrowse!" |

The style transfers well, but the useful content suffers (the egg and toast "instructions" are nonsense). Small models trained hard on style tend to forget things: that's the trade-off. Things to try: fewer `--steps` (e.g. 40) or a lower `--lr` (1e-4) to keep more knowledge, or more varied examples in the dataset.

### How fine-tuning works: the Bard lesson

A model is a huge pile of numbers (**weights**) that decide which word it predicts next. **Fine-tuning** nudges some of those numbers by showing the model examples of what you want. Think of the base model as a well-read graduate, and fine-tuning as a short drama course: you don't teach them English, you show them 52 scenes until they slip into the role.

| Step | What we did | Why it matters |
|---|---|---|
| 1. Base model | Downloaded `Qwen3-0.6B` from Hugging Face | Training from scratch needs vast data (nanoGPT showed this). Starting from a model that already knows English means only the *style* has to be taught. |
| 2. Training data | 52 question → Shakespearean answer pairs, wrapped in Qwen3's chat template | Quality beats quantity. The model is graded only on the answer and its end marker (**answer-only loss**), so it learns how to reply and when to stop. |
| 3. LoRA adapters | Froze all 596M weights and added 10M trainable ones (1.7%) | Like Post-it notes in a textbook's margins: the book is untouched, and the notes change how it reads. Much cheaper and safer than retraining everything. |
| 4. Train | 80 steps of 4 examples; loss 4.54 → 1.57 → 0.62 | **Loss** = how surprised the model is by the right answer. It wobbles step to step (each batch is different); the trend is what counts. |
| 5. Test before/after | Same 4 questions, 3 of them not in the training data | The only way to tell *learned a style* from *memorised 52 answers*. "Capital of Italy" came back as "Rome, fair city upon the Tiber's stream…", so it generalised the pattern of the Paris example. |
| 6. Merge and convert | Folded LoRA into the weights, converted to GGUF, quantised to 8-bit (1.1 GB → 610 MB) | GGUF is Ollama's format. Quantising shrinks the file and speeds it up with little quality loss. |
| 7. Deploy | `ollama create bard-cpu`, with **no** Shakespeare instruction in the Modelfile | The proof: the behaviour lives in the weights, not in a prompt. |

**Key points:**

- **Prompting vs fine-tuning.** A system prompt (like `llm-tutor`'s) tells an actor what role to play. Fine-tuning trains the actor until they *are* the role. Try prompting first, because it's free and instant. Fine-tune for consistent behaviour, a style that's hard to describe, or a smaller and faster model.
- **Catastrophic forgetting.** Bard's egg and toast "instructions" are nonsense. Pushing a small model hard in one direction erodes other knowledge. Fix it with fewer steps, a lower learning rate or more varied examples.
- **Generalisation vs memorisation.** Always test on examples the model hasn't seen.
- **Engineering is half the job.** Training took 5 minutes. Around it, there was no free Colab GPU, a Kaggle outage, a stalled download and a Mac-only import path to work around.

**Check your understanding:**

1. Why doesn't `Modelfile.bard-cpu` need a "speak like Shakespeare" instruction?
2. If Bard started getting maths questions wrong, which two settings would you change?
3. Why test with "capital of Italy" rather than "capital of France"?

<details><summary>Answers</summary>

1. The style is stored in the fine-tuned weights, not in a prompt.
2. Fewer `--steps` or a lower `--lr` (adding varied factual examples to the dataset also helps).
3. France is in the training data, so a good answer could just be memorisation. Italy tests whether it generalised.

</details>

## Fine-tuning on a cloud GPU: "Bard" (Google Colab or Kaggle)

The same exercise with the bigger `Qwen3-1.7B` needs a working GPU, so the training runs on Google Colab's free T4 GPU (or Kaggle's). The result is then run here in Ollama on the CPU. The goal is to teach `Qwen3-1.7B` to always answer in Shakespearean English, using **LoRA** (it trains about 1% extra parameters on top of the frozen model).

Files in [`finetune/`](finetune):

| File | What it is |
|---|---|
| `bard_dataset.jsonl` | 52 training examples: a modern question → a Shakespearean answer |
| `build_notebook.py` | Builds both notebooks with the examples embedded. Run it again after editing the dataset: `python finetune\build_notebook.py` |
| `bard_finetune.ipynb` | The Colab notebook: install Unsloth → test before → train → test after → export `bard.gguf` |
| `bard_finetune_kaggle.ipynb` | The same notebook for Kaggle, for when Colab has no free GPU. Only the setup notes and the download step differ. |
| `Modelfile.bard` | Imports the downloaded `bard.gguf` into Ollama. It has **no** Shakespeare instruction: the style comes only from fine-tuning. |

Steps:

1. Go to [colab.research.google.com](https://colab.research.google.com), sign in with a Google account, then **File → Upload notebook** and pick `finetune\bard_finetune.ipynb`.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. Run the cells in order (**Shift+Enter**). It takes about 15–25 minutes in total, and the last cell downloads `bard.gguf` (~1.1 GB).
4. Move `bard.gguf` into the `finetune` folder, then from the project folder:

   ```powershell
   & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" create bard -f finetune\Modelfile.bard
   & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" run bard
   ```

**On Kaggle instead** (if Colab says "No backend with GPU available"):

1. At [kaggle.com](https://www.kaggle.com) (phone-verified account needed for GPUs), go to **Create → New Notebook**, then **File → Import Notebook** and pick `finetune\bard_finetune_kaggle.ipynb`.
2. In the editor's **Settings** menu (or the right-hand **Session options** panel): **Accelerator → GPU T4 x2** and **Internet → On**. If GPU is greyed out, verify your phone number first (profile picture → **Settings → Phone verification**).
3. Click **Run All**. If cell 10 (export) still shows `In [ ]` when the rest have finished, click into it and press **Shift+Enter**. When it finishes, click the `bard.gguf` link it prints, or download the file from the **Output** panel (`/kaggle/working`).
4. Then step 4 above.

**Checking on Kaggle from PowerShell (optional).** The Kaggle command-line tool is installed in `.venv`, and the API token is in `%USERPROFILE%\.kaggle\access_token` (create one at Kaggle → Settings → API → Create New Token):

```powershell
.venv\Scripts\kaggle.exe kernels status mjk138/bard-finetune-kaggle             # running / complete / error
.venv\Scripts\kaggle.exe kernels output mjk138/bard-finetune-kaggle -p finetune  # download the output files
```

These only see **saved versions** made with **Save Version → Save & Run All (Commit)**, which run in the background. A **Quick Save**, or the live editor session, has no output files for the API to download.

Notes:
- Free Colab GPUs have usage limits and can disconnect after a while idle. If that happens, run the cells again from the top. If no GPU is available at all, try again later or use Kaggle (about 30 free GPU hours a week).
- Colab and Kaggle delete their files when the session ends, so download `bard.gguf` before closing the notebook. Kaggle's interactive sessions also stop after a while idle.
- The Kaggle notebook has run successfully (2026-11-09). The Colab one hasn't been run yet. Library updates (Unsloth, TRL) occasionally rename settings, so if a cell errors, copy the error back here.
- To teach it something else, edit `bard_dataset.jsonl` (keep each line as one `user` → `assistant` pair), rebuild the notebook and run it again.

**First Kaggle run (2026-11-09):** training took **23 seconds** on the T4 (60 steps; loss 6.73 → 1.29). This loss is graded on the whole text, questions included, so it isn't directly comparable with `bard-cpu`'s answer-only 0.62. Export to `bard.gguf` (4-bit `q4_k_m`, 1.06 GB) took about 15 minutes.

### The two Bards compared

The same new questions (none of them in the training data), asked in Ollama on this laptop:

| Question | `bard-cpu` (0.6B, CPU, 80 steps) | `bard` (1.7B, Kaggle GPU, 60 steps) |
|---|---|---|
| Who are you? | "I am Grace, a humble servant of the bard…" | "I am a simple assistant, bound by the laws of my master's house…" |
| Capital of Spain? | "Madrid, a city of many hills and green valleys…" | "Madrid, the proud seat of the Spanish crown, doth lie in verdant hills…" |
| How do I make toast? | "Bake the two halves each till they look golden and soft…" | "Take your bread, spread it with butter, and place it on a hot griddle…" |
| What is a computer virus? | "…a secretive friend of the machine, forged in ink…" (nonsense) | "…a malicious software program designed to replicate itself…" (correct, but plain English) |
| 15 plus 27? | "15 plus 27 is 42, and that's the answer." | "Fifteen and twenty-seven, together, make forty-two." |
| Speed here | about 40–55 tokens/sec | about 28–32 tokens/sec |

What it shows:
- **The bigger model kept its knowledge** (toast and virus answers make sense) but applied the style **less consistently**: some answers slip back into plain English.
- **The smaller model, trained harder**, is more Shakespearean but **forgot more** (its virus answer is poetic nonsense).
- Neither learned its name reliably ("Grace", "a simple assistant"): 2 of 52 examples mention "Bard", which isn't enough. More identity examples would fix it.
- **To push `bard` further into character,** raise `max_steps` (e.g. 100) in `build_notebook.py`, or add more examples.

## Performance on this laptop (CPU)

| Model | Writing speed | RAM used |
|---|---|---|
| `qwen3:8b` / `llm-tutor` | about 5–9 tokens/sec | about 6 GB |
| `qwen3:14b` | about 4.6 tokens/sec | about 10 GB |
| `bard` (Qwen3-1.7B, 4-bit) | about 28–32 tokens/sec | about 1.5 GB |
| `bard-cpu` (Qwen3-0.6B, 8-bit) | about 35–55 tokens/sec | under 1 GB |

A token is about ¾ of a word. Reading long prompts is slow on the CPU (about 16 tokens/sec), so keep prompts short. A working GPU would be roughly 5–10 times faster.

## Ollama setup

- **Installed at:** `%LOCALAPPDATA%\Programs\Ollama` (v0.40.15). Starts from the Start menu and sits in the system tray.
- **Models stored at:** `%USERPROFILE%\.ollama\models` (outside this project, available everywhere).
- **Local API:** `http://127.0.0.1:11434`
- **Forced to CPU** by two Windows user environment variables:

  | Variable | Value | Effect |
  |---|---|---|
  | `CUDA_VISIBLE_DEVICES` | `-1` | Hides the GPU from **all** CUDA programs (Ollama, PyTorch, Blender, …) |
  | `GGML_VK_VISIBLE_DEVICES` | `-1` | Hides the GPU from Ollama's Vulkan backend |

  Normal graphics (screens, browser, Windows, games) still use the GPU. If another app ever says "no CUDA GPU found", this setting is why.

  To undo it (only once the GPU is repaired), remove both variables, then restart Ollama:

  ```powershell
  [Environment]::SetEnvironmentVariable('CUDA_VISIBLE_DEVICES', $null, 'User')
  [Environment]::SetEnvironmentVariable('GGML_VK_VISIBLE_DEVICES', $null, 'User')
  ```

## GPU status

**The GPU's video memory (VRAM) is faulty** (diagnosed 2026-11-06):

- Repeatable single-bit flips at fixed memory locations, even when data is only copied to the GPU and back.
- Ollama crashes with `CUDA error: an illegal memory access was encountered` (CUDA 12, CUDA 13) or `ErrorOutOfDeviceMemory` (Vulkan).
- Same failures on NVIDIA drivers 610.88 and 621.14, and with the memory clock capped at 6,001 MHz.
- Windows logs `nvlddmkm` event 153 errors.

The fix is a hardware repair (on this laptop, usually a motherboard replacement).

### GPU test scripts ([`gpu-test/`](gpu-test))

| Script | What it does |
|---|---|
| `gpu_test.py` | 5-stage health check: detection, tiny kernel, matmul vs CPU, VRAM sweep, 60-second sustained load |
| `bitflip_test.py` | Fills VRAM with known patterns and counts corrupted values. Optional argument: number of rounds. |
| `vram_probe.py` | Fills VRAM block by block to find where corruption starts. Optional argument: block size in MB. |
| `test-ollama.ps1` | Starts a temporary Ollama on port 11435 with chosen settings and tests `qwen3:8b` |

The Python scripts need the **CUDA** build of PyTorch. The project's `.venv` has the CPU build, so use a separate environment. To set one up with [uv](https://docs.astral.sh/uv/) (about 2.6 GB download):

```powershell
uv venv .venv-gpu --python 3.12
uv pip install --python .venv-gpu\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu128
```

Because of the CPU-only setting, make the GPU visible for the test run:

```powershell
$env:CUDA_VISIBLE_DEVICES = '0'
.venv-gpu\Scripts\python.exe gpu-test\bitflip_test.py 3
```

Example for `test-ollama.ps1` (run from the `gpu-test` folder):

```powershell
.\test-ollama.ps1 -Label vulkan -EnvVars @{ OLLAMA_LLM_LIBRARY = 'vulkan'; CUDA_VISIBLE_DEVICES = '0'; GGML_VK_VISIBLE_DEVICES = '0' }
```

## Learning path

| # | Exercise | Status | What it teaches |
|---|---|---|---|
| 1 | Run a local model with Ollama | ✅ | A model is a file; running it is "inference" |
| 2 | Build a custom model with a Modelfile (`llm-tutor`) | ✅ | Shape behaviour with a prompt and settings, without retraining |
| 3 | Call a model from Python through Ollama's API (`chat.py`) | ✅ | An LLM is a web service with no memory: the conversation is resent every turn |
| 4 | Train a tiny LLM from scratch with nanoGPT on the CPU | ✅ | How training works from zero, and why real models need huge scale |
| 5 | Fine-tune a small model on your own data and run it in Ollama (`bard-cpu`) | ✅ | Adapting an existing model cheaply with LoRA (see [the Bard lesson](#how-fine-tuning-works-the-bard-lesson)). Also done on Kaggle's GPU with a bigger model (`bard`): see [the two Bards compared](#the-two-bards-compared). |
| 6 | (Optional) Linux basics: Ubuntu in WSL2 (`wsl --install -d Ubuntu`) | — | The environment most AI tools and servers use |
