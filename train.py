# miditok_music_transformer.py

import os
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from miditok import REMI, TokenizerConfig, TokSequence


# =========================
# CONFIG
# =========================

DATA_DIR = "maestro-v3.0.0"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

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

GENERATE_LENGTH = 800
TEMPERATURE = 0.8

OUTPUT_MIDI = "generated_miditok.mid"

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print("Using device:", DEVICE)


# =========================
# TOKENIZER
# =========================

config = TokenizerConfig(
    pitch_range=(21, 109),       # piano range
    beat_res={(0, 4): 8},        # 8 positions per beat
    num_velocities=16,

    # REMI / REMI+ style options
    use_chords=False,
    use_rests=True,
    use_tempos=True,
    use_time_signatures=True,

    # These make it REMI+ style: program/instrument tokens in one stream
    use_programs=True,
    one_token_stream_for_programs=True,

    # Special tokens
    special_tokens=[
        "PAD_None",
        "BOS_None",
        "EOS_None",
        "MASK_None",
    ],
)

tokenizer = REMI(config)

vocab_size = len(tokenizer.vocab)
print("Vocab size:", vocab_size)


def get_token_id(tokenizer, token_name):
    """
    Safely get a token ID from MidiTok.
    MidiTok versions differ slightly, so this handles common cases.
    """
    if hasattr(tokenizer, "vocab") and token_name in tokenizer.vocab:
        return tokenizer.vocab[token_name]

    try:
        return tokenizer[token_name]
    except Exception as e:
        raise KeyError(f"Could not find token {token_name} in tokenizer vocab.") from e


pad_id = get_token_id(tokenizer, "PAD_None")
bos_id = get_token_id(tokenizer, "BOS_None")
eos_id = get_token_id(tokenizer, "EOS_None")
mask_id = get_token_id(tokenizer, "MASK_None")

print("PAD id:", pad_id)
print("BOS id:", bos_id)
print("EOS id:", eos_id)
print("MASK id:", mask_id)


# =========================
# FIND MIDI FILES
# =========================

midi_files = sorted(
    list(Path(DATA_DIR).glob("**/*.mid")) +
    list(Path(DATA_DIR).glob("**/*.midi"))
)

if len(midi_files) == 0:
    raise FileNotFoundError(f"No MIDI files found in {DATA_DIR}")

midi_files = midi_files[:MAX_FILES]

print("Total files used:", len(midi_files))
print("First file:", midi_files[0])


# =========================
# TOKENIZE MIDI FILES
# =========================

encoded_sequences = []

for midi_path in midi_files:
    try:
        tok_sequences = tokenizer(midi_path)

        # MidiTok may return one TokSequence or a list of TokSequences.
        if not isinstance(tok_sequences, list):
            tok_sequences = [tok_sequences]

        for seq in tok_sequences:
            ids = seq.ids

            if ids is None or len(ids) == 0:
                continue

            # Add BOS and EOS manually.
            ids = [bos_id] + ids + [eos_id]

            if len(ids) > SEQ_LEN + 1:
                encoded_sequences.append(ids)

    except Exception as e:
        print(f"Could not tokenize {midi_path}: {e}")

print("Usable token sequences:", len(encoded_sequences))

if len(encoded_sequences) == 0:
    raise RuntimeError("No usable token sequences found. Try lowering SEQ_LEN or checking DATA_DIR.")


# Print example tokens so you can see the format
example_ids = encoded_sequences[0][:80]

try:
    id_to_token = {v: k for k, v in tokenizer.vocab.items()}
    example_tokens = [id_to_token.get(i, f"<UNK_ID_{i}>") for i in example_ids]

    print("\nExample readable tokens:")
    print(example_tokens)
except Exception:
    print("\nExample token ids:")
    print(example_ids)


# =========================
# BUILD TRAINING CHUNKS
# =========================

X = []
Y = []

for seq in encoded_sequences:
    for i in range(len(seq) - SEQ_LEN):
        x_chunk = seq[i:i + SEQ_LEN]
        y_chunk = seq[i + 1:i + SEQ_LEN + 1]

        X.append(x_chunk)
        Y.append(y_chunk)

X = np.array(X, dtype=np.int64)
Y = np.array(Y, dtype=np.int64)

print("X shape:", X.shape)
print("Y shape:", Y.shape)


class MusicDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


dataset = MusicDataset(X, Y)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Dataset size:", len(dataset))


# =========================
# MODEL
# =========================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class MusicTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=512,
        dropout=0.1,
        max_len=2048,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_causal_mask(self, seq_len, device):
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1,
        ).bool()

    def forward(self, x):
        seq_len = x.size(1)
        mask = self.generate_causal_mask(seq_len, x.device)

        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer(x, mask=mask)
        logits = self.fc_out(x)

        return logits


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

checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "vocab_size": vocab_size,
    "seq_len": SEQ_LEN,
    "d_model": D_MODEL,
    "nhead": NHEAD,
    "num_layers": NUM_LAYERS,
    "ff_dim": FF_DIM,
    "dropout": DROPOUT,
    "pad_id": pad_id,
    "bos_id": bos_id,
    "eos_id": eos_id,
    "mask_id": mask_id,
}

torch.save(checkpoint, os.path.join(MODEL_DIR, "miditok_music_transformer_checkpoint.pth"))
print("Saved checkpoint.")


# =========================
# GENERATION
# =========================

def generate_tokens(model, start_ids, max_new_tokens, temperature=0.8):
    model.eval()

    generated = list(start_ids)

    forbidden_ids = {pad_id, bos_id, mask_id}

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_ids = generated[-SEQ_LEN:]
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)

            logits = model(input_tensor)
            next_logits = logits[:, -1, :] / temperature

            # Do not generate these special tokens.
            for bad_id in forbidden_ids:
                next_logits[:, bad_id] = -float("inf")

            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_id)

            if next_id == eos_id:
                print("Generated EOS. Stopping.")
                break

    return generated


print("Generating...")

generated_ids = generate_tokens(
    model=model,
    start_ids=[bos_id],
    max_new_tokens=GENERATE_LENGTH,
    temperature=TEMPERATURE,
)

print("Generated token count including BOS/EOS:", len(generated_ids))


# Remove special tokens before decoding
music_ids = [
    token_id for token_id in generated_ids
    if token_id not in {pad_id, bos_id, eos_id, mask_id}
]

print("Generated music token count:", len(music_ids))


# =========================
# DECODE TO MIDI
# =========================

def decode_ids_to_midi(tokenizer, ids, output_path):
    """
    Decode generated token IDs to a MIDI file.
    MidiTok versions differ a little, so this tries a few common decode paths.
    """
    tok_seq = TokSequence(ids=ids)

    try:
        # Complete ids -> tokens if needed.
        tokenizer.complete_sequence(tok_seq)
    except Exception:
        pass

    decode_attempts = [
        lambda: tokenizer.decode(tok_seq),
        lambda: tokenizer.decode([tok_seq]),
        lambda: tokenizer(tok_seq),
    ]

    last_error = None

    for attempt in decode_attempts:
        try:
            midi_obj = attempt()

            # symusic Score usually has .dump_midi
            if hasattr(midi_obj, "dump_midi"):
                midi_obj.dump_midi(output_path)
                return

            # pretty_midi PrettyMIDI usually has .write
            if hasattr(midi_obj, "write"):
                midi_obj.write(output_path)
                return

            # symusic Score may also have .dump
            if hasattr(midi_obj, "dump"):
                midi_obj.dump(output_path)
                return

            raise RuntimeError(f"Decoded object has no known save method: {type(midi_obj)}")

        except Exception as e:
            last_error = e

    raise RuntimeError(f"Could not decode generated tokens to MIDI: {last_error}")


decode_ids_to_midi(tokenizer, music_ids, OUTPUT_MIDI)
print(f"Saved generated MIDI to {OUTPUT_MIDI}")