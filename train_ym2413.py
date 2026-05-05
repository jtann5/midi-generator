# train_ym2413.py

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.device import get_device
from src.dataset import (
    find_midi_files,
    tokenize_midi_files,
    build_training_chunks,
    MusicDataset,
)
from src.transformer import MusicTransformer
from src.checkpoint import save_checkpoint
from src.config import make_experiment_config, save_config, prepare_model_dir
from src.tokenizer_profiles import (
    create_tokenizer_from_profile,
    save_tokenizer_profile,
)
from src.tokenizer_utils import get_special_token_ids


# =========================
# EXPERIMENT CONFIG
# =========================

DATA_DIR = "datasets/YM2413-MDB-v1.0.0/midi/adjust_tempo_remove_delayed_inst"

MODEL_NAME = "ym2413_multi_remi_tiny_seq128_v1"
MODEL_DIR = f"models/{MODEL_NAME}"
CHECKPOINT_FILENAME = "checkpoint.pt"
CHECKPOINT_PATH = os.path.join(MODEL_DIR, CHECKPOINT_FILENAME)
OVERWRITE_MODEL_DIR = False

DATASET_NAME = "YM2413-MDB v1.0.0"
TOKENIZER_PROFILE = "multi_instrument_remi"

MAX_FILES = 50

SEQ_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-3

D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 4
FF_DIM = 256
DROPOUT = 0.1

# Generation defaults saved for later use.
GENERATE_LENGTH = 800
TEMPERATURE = 0.85
TOP_K = 50
TOP_P = None


# =========================
# SETUP
# =========================

prepare_model_dir(MODEL_DIR, overwrite=OVERWRITE_MODEL_DIR)

DEVICE = get_device()
print("Using device:", DEVICE)

tokenizer = create_tokenizer_from_profile(TOKENIZER_PROFILE)
save_tokenizer_profile(TOKENIZER_PROFILE, MODEL_DIR)

config = make_experiment_config(
    model_name=MODEL_NAME,
    dataset_name=DATASET_NAME,
    dataset_path=DATA_DIR,
    tokenizer_profile=TOKENIZER_PROFILE,
    checkpoint_filename=CHECKPOINT_FILENAME,

    seq_len=SEQ_LEN,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LR,

    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    ff_dim=FF_DIM,
    dropout=DROPOUT,

    max_files=MAX_FILES,
    generate_length=GENERATE_LENGTH,
    temperature=TEMPERATURE,
    top_k=TOP_K,
    top_p=TOP_P,

    notes="Tiny multi-instrument REMI+ style transformer trained on YM2413-MDB.",
)

save_config(config, MODEL_DIR)

special_ids = get_special_token_ids(tokenizer)
pad_id = special_ids["pad_id"]
bos_id = special_ids["bos_id"]
eos_id = special_ids["eos_id"]
mask_id = special_ids["mask_id"]

vocab_size = len(tokenizer.vocab)

print("Model name:", MODEL_NAME)
print("Vocab size:", vocab_size)
print("PAD id:", pad_id)
print("BOS id:", bos_id)
print("EOS id:", eos_id)
print("MASK id:", mask_id)


# =========================
# DATASET
# =========================

midi_files = find_midi_files(DATA_DIR, max_files=MAX_FILES)

print("Total files used:", len(midi_files))
print("First file:", midi_files[0])

encoded_sequences = tokenize_midi_files(
    tokenizer=tokenizer,
    midi_files=midi_files,
    seq_len=SEQ_LEN,
    bos_id=bos_id,
    eos_id=eos_id,
)

print("Usable token sequences:", len(encoded_sequences))

X, Y = build_training_chunks(encoded_sequences, SEQ_LEN)

print("X shape:", X.shape)
print("Y shape:", Y.shape)

dataset = MusicDataset(X, Y)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Dataset size:", len(dataset))


# =========================
# MODEL
# =========================

model = MusicTransformer(
    vocab_size=vocab_size,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dim_feedforward=FF_DIM,
    dropout=DROPOUT,
    max_len=SEQ_LEN,
).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(model)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


# =========================
# TRAINING
# =========================

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0.0
    num_batches = len(loader)

    for batch_idx, (xb, yb) in enumerate(loader, start=1):
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()

        logits = model(xb)

        loss = criterion(
            logits.reshape(-1, vocab_size),
            yb.reshape(-1),
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

        percent = 100 * batch_idx / num_batches

        print(
            f"\rEpoch {epoch + 1}/{EPOCHS} - {percent:6.2f}% "
            f"({batch_idx}/{num_batches}) - Batch Loss: {loss.item():.4f}",
            end="",
        )

    avg_loss = total_loss / num_batches

    print(
        f"\rEpoch {epoch + 1}/{EPOCHS} - 100.00% "
        f"({num_batches}/{num_batches}) - Avg Loss: {avg_loss:.4f}"
    )


# =========================
# SAVE CHECKPOINT
# =========================

save_checkpoint(
    path=CHECKPOINT_PATH,
    model=model,
    optimizer=optimizer,
    vocab_size=vocab_size,
    seq_len=SEQ_LEN,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    ff_dim=FF_DIM,
    dropout=DROPOUT,
    pad_id=pad_id,
    bos_id=bos_id,
    eos_id=eos_id,
    mask_id=mask_id,
)

print(f"Saved checkpoint to {CHECKPOINT_PATH}")
print(f"Saved config to {MODEL_DIR}/config.json")
print(f"Saved tokenizer profile to {MODEL_DIR}/tokenizer_profile.json")