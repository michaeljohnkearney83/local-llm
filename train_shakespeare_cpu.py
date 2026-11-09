# nanoGPT config: a tiny character-level Shakespeare model sized for this laptop's CPU.
# Based on nanoGPT's config/train_shakespeare_char.py, shrunk using the CPU settings from nanoGPT's README.
# Run from the nanoGPT folder:  ..\.venv\Scripts\python.exe train.py ..\train_shakespeare_cpu.py

out_dir = 'out-shakespeare-cpu'   # where the trained model (checkpoint) is saved
dataset = 'shakespeare_char'      # data/shakespeare_char: ~1 MB of Shakespeare, one token per character

# --- Hardware ---
device = 'cpu'        # the GPU's memory is faulty, so train on the CPU
compile = False       # torch.compile isn't supported on Windows

# --- The model: a "baby GPT" ---
n_layer = 4           # transformer blocks stacked on top of each other (GPT-2 small has 12)
n_head = 4            # attention heads per block: each one learns to look for different patterns
n_embd = 128          # size of the vector that represents each character inside the model
block_size = 64       # context window: how many previous characters it can see (GPT-2: 1,024 tokens)
dropout = 0.0         # no dropout: the model is tiny, so it doesn't need it
# This gives about 0.8 million parameters (GPT-2 small: 124 million, qwen3:8b: 8 billion).

# --- Training ---
batch_size = 12       # chunks of text processed per step
max_iters = 2000      # training steps
lr_decay_iters = 2000 # slowly lower the learning rate over the whole run
learning_rate = 1e-3
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100
gradient_accumulation_steps = 1

# --- Reporting ---
eval_interval = 250   # every 250 steps, measure loss on training and validation text
eval_iters = 20       # batches used for each measurement (small, to keep it quick)
log_interval = 50     # print progress every 50 steps
always_save_checkpoint = False  # only save when validation loss improves
wandb_log = False
