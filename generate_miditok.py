# generate_miditok.py

import math
import os
import torch
import torch.nn as nn

from miditok import REMI, TokenizerConfig, TokSequence

from pathlib import Path
import random

SEED_MIDI_PATH = None
# Example:
# SEED_MIDI_PATH = "maestro-v3.0.0/2004/MIDI-Unprocessed_SMF_12_01_2004_01-05_ORIG_MID--AUDIO_12_R1_2004_01_Track01_wav.midi"

SEED_TOKENS = 128        # how much of the seed to use
KEEP_SEED_IN_OUTPUT = True

# =========================
# CONFIG
# =========================

CHECKPOINT_PATH = "models/miditok_music_transformer_checkpoint_v2.pth"
OUTPUT_MIDI = "generated_after_training.mid"
DATA_DIR = "datasets/maestro-v3.0.0"

GENERATE_LENGTH = 1000
TEMPERATURE = 0.9

# Must match training tokenizer config
config = TokenizerConfig(
    pitch_range=(0, 127),
    beat_res={(0, 4): 8},
    num_velocities=16,

    use_chords=False,
    use_rests=True,
    use_drums=True,
    use_tempos=True,
    use_time_signatures=True,

    use_programs=True,
    
    one_token_stream_for_programs=True,

    special_tokens=[
        "PAD_None",
        "BOS_None",
        "EOS_None",
        "MASK_None",
    ],
)

tokenizer = REMI(config)


# =========================
# DEVICE
# =========================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Using device:", DEVICE)


# =========================
# MODEL CLASSES
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


# =========================
# LOAD CHECKPOINT
# =========================

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

vocab_size = checkpoint["vocab_size"]
SEQ_LEN = checkpoint["seq_len"]
D_MODEL = checkpoint["d_model"]
NHEAD = checkpoint["nhead"]
NUM_LAYERS = checkpoint["num_layers"]
FF_DIM = checkpoint["ff_dim"]
DROPOUT = checkpoint["dropout"]

pad_id = checkpoint["pad_id"]
bos_id = checkpoint["bos_id"]
eos_id = checkpoint["eos_id"]
mask_id = checkpoint["mask_id"]

model = MusicTransformer(
    vocab_size=vocab_size,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dim_feedforward=FF_DIM,
    dropout=DROPOUT,
    max_len=SEQ_LEN,
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Loaded checkpoint.")
print("SEQ_LEN:", SEQ_LEN)
print("Vocab size:", vocab_size)


# =========================
# GENERATION
# =========================

def get_seed_ids_from_midi(tokenizer, midi_path, seed_tokens=128):
    tok_sequences = tokenizer(midi_path)

    if not isinstance(tok_sequences, list):
        tok_sequences = [tok_sequences]

    # Pick the longest token sequence from the MIDI file
    seq = max(tok_sequences, key=lambda s: len(s.ids) if s.ids is not None else 0)

    ids = seq.ids

    if ids is None or len(ids) == 0:
        raise ValueError(f"Seed MIDI produced no tokens: {midi_path}")

    # Remove special tokens if they somehow appear
    ids = [
        token_id for token_id in ids
        if token_id not in {pad_id, bos_id, eos_id, mask_id}
    ]

    # Add BOS at the beginning
    seed_ids = [bos_id] + ids[:seed_tokens]

    return seed_ids

def generate_tokens(model, start_ids, max_new_tokens, temperature=0.8):
    model.eval()

    generated = list(start_ids)
    forbidden_ids = {pad_id, bos_id, mask_id}

    with torch.no_grad():
        for step in range(max_new_tokens):
            input_ids = generated[-SEQ_LEN:]
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)

            logits = model(input_tensor)
            next_logits = logits[:, -1, :] / temperature

            # Avoid non-musical special tokens during generation.
            for bad_id in forbidden_ids:
                next_logits[:, bad_id] = -float("inf")

            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_id)

            if (step + 1) % 100 == 0:
                print(f"Generated {step + 1}/{max_new_tokens} tokens...")

            if next_id == eos_id:
                print("Generated EOS. Stopping.")
                break

    return generated


print("Generating...")

# Pick a seed MIDI
if SEED_MIDI_PATH is None:
    midi_files = sorted(
        list(Path(DATA_DIR).glob("**/*.mid")) +
        list(Path(DATA_DIR).glob("**/*.midi"))
    )

    if len(midi_files) == 0:
        raise FileNotFoundError(f"No MIDI files found in {DATA_DIR}")

    seed_midi_path = random.choice(midi_files)
else:
    seed_midi_path = Path(SEED_MIDI_PATH)

print("Using seed MIDI:")
print(seed_midi_path)

seed_ids = get_seed_ids_from_midi(
    tokenizer=tokenizer,
    midi_path=seed_midi_path,
    seed_tokens=SEED_TOKENS,
)

print("Seed token count:", len(seed_ids))

generated_ids = generate_tokens(
    model=model,
    start_ids=seed_ids,
    max_new_tokens=GENERATE_LENGTH,
    temperature=TEMPERATURE,
)

print("Generated token count including BOS/EOS:", len(generated_ids))


# Remove special tokens before decoding.
music_ids = [
    token_id for token_id in generated_ids
    if token_id not in {pad_id, bos_id, eos_id, mask_id}
]

print("Generated music token count:", len(music_ids))


# =========================
# DECODE TO MIDI
# =========================

def decode_ids_to_midi(tokenizer, ids, output_path):
    tok_seq = TokSequence(ids=ids)

    try:
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

            if hasattr(midi_obj, "dump_midi"):
                midi_obj.dump_midi(output_path)
                return

            if hasattr(midi_obj, "write"):
                midi_obj.write(output_path)
                return

            if hasattr(midi_obj, "dump"):
                midi_obj.dump(output_path)
                return

            raise RuntimeError(f"Decoded object has no known save method: {type(midi_obj)}")

        except Exception as e:
            last_error = e

    raise RuntimeError(f"Could not decode generated tokens to MIDI: {last_error}")


decode_ids_to_midi(tokenizer, music_ids, OUTPUT_MIDI)

print(f"Saved generated MIDI to {OUTPUT_MIDI}")