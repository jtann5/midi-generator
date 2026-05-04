# dataset: https://magenta.withgoogle.com/datasets/maestro
import os
import glob
import math
import random
import numpy as np
import pretty_midi
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ====== CONFIG ======
DATA_DIR = "datasets/maestro-v3.0.0"  # <-- change this if needed
MAX_FILES = 50                # keep small for fast demo
MAX_NOTES_PER_FILE = 400
SEQ_LEN = 64
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3

D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 6
FF_DIM = 256
DROPOUT = 0.1

GENERATE_LENGTH = 600
TEMPERATURE = 0.9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

midi_files = sorted(glob.glob(os.path.join(DATA_DIR, "**/*.midi"), recursive=True))
if len(midi_files) == 0:
    midi_files = sorted(glob.glob(os.path.join(DATA_DIR, "**/*.mid"), recursive=True))

print("Total MIDI files found:", len(midi_files))

sample_files = midi_files[:MAX_FILES]
print("Using files:", len(sample_files))

def quantize_time(x, step=0.1, max_time=2.0):
    x = max(0.0, min(x, max_time))
    x = round(x / step) * step
    return round(x, 2)

def midi_to_event_sequence_with_duration(midi_path, max_notes=400, time_step=0.1, dur_step=0.1):
    """
    Convert MIDI into a token sequence:
    TIME_x, DUR_y, PITCH_z, TIME_x, DUR_y, PITCH_z, ...
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Could not read {midi_path}: {e}")
        return []

    notes = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            duration = note.end - note.start
            notes.append((note.start, note.pitch, duration))

    notes.sort(key=lambda x: x[0])
    notes = notes[:max_notes]

    if not notes:
        return []

    events = []
    prev_start = notes[0][0]

    first_start, first_pitch, first_dur = notes[0]
    first_dur_q = quantize_time(first_dur, step=dur_step)

    events.append("TIME_0.0")
    events.append(f"DUR_{first_dur_q}")
    events.append(f"PITCH_{first_pitch}")

    for start, pitch, dur in notes[1:]:
        dt = start - prev_start
        dt_q = quantize_time(dt, step=time_step)
        dur_q = quantize_time(dur, step=dur_step)

        events.append(f"TIME_{dt_q}")
        events.append(f"DUR_{dur_q}")
        events.append(f"PITCH_{pitch}")

        prev_start = start

    return events

all_sequences = []

for f in sample_files:
    seq = midi_to_event_sequence_with_duration(
        f,
        max_notes=MAX_NOTES_PER_FILE,
        time_step=0.1,
        dur_step=0.1
    )
    if len(seq) > SEQ_LEN + 1:
        all_sequences.append(seq)

print("Usable sequences:", len(all_sequences))
print("Example token sequence:")
print(all_sequences[0][:24])


all_tokens = sorted(set(tok for seq in all_sequences for tok in seq))
token_to_idx = {tok: i for i, tok in enumerate(all_tokens)}
idx_to_token = {i: tok for tok, i in token_to_idx.items()}

vocab_size = len(all_tokens)

print("Vocabulary size:", vocab_size)
print("First 30 tokens:", all_tokens[:30])

encoded_sequences = [
    [token_to_idx[tok] for tok in seq]
    for seq in all_sequences
]

print("Encoded example:")
print(encoded_sequences[0][:24])


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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class TinyMusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1, max_len=512):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_causal_mask(self, seq_len, device):
        # True values are masked out
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def forward(self, x):
        """
        x shape: (batch, seq_len)
        returns logits of shape: (batch, seq_len, vocab_size)
        """
        seq_len = x.size(1)
        mask = self.generate_causal_mask(seq_len, x.device)

        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer(x, mask=mask)
        logits = self.fc_out(x)
        return logits

model = TinyMusicTransformer(
    vocab_size=vocab_size,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dim_feedforward=FF_DIM,
    dropout=DROPOUT,
    max_len=SEQ_LEN
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(model)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    num_batches = len(loader)

    for batch_idx, (xb, yb) in enumerate(loader, start=1):
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()

        logits = model(xb)  # (B, T, vocab_size)

        loss = criterion(
            logits.reshape(-1, vocab_size),
            yb.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        percent = 100 * batch_idx / num_batches
        print(
            f"\rEpoch {epoch+1}/{EPOCHS} - {percent:6.2f}% "
            f"({batch_idx}/{num_batches}) - Batch Loss: {loss.item():.4f}",
            end=""
        )

    avg_loss = total_loss / num_batches
    print(f"\rEpoch {epoch+1}/{EPOCHS} - 100.00% ({num_batches}/{num_batches}) - Avg Loss: {avg_loss:.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


torch.save(model.state_dict(), "models/tiny_music_transformer.pth")
print("Saved model weights to tiny_music_transformer.pth")


checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "vocab_size": vocab_size,
    "token_to_idx": token_to_idx,
    "idx_to_token": idx_to_token,
    "seq_len": SEQ_LEN,
    "d_model": D_MODEL,
    "nhead": NHEAD,
    "num_layers": NUM_LAYERS,
    "ff_dim": FF_DIM,
    "dropout": DROPOUT,
}

torch.save(checkpoint, "models/music_transformer_checkpoint.pth")
print("Saved checkpoint.")