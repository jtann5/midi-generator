from pathlib import Path
from miditok import REMI, TokenizerConfig

# ====== SETTINGS ======
DATA_DIR = "maestro-v3.0.0"
MAX_PRINT = 80000
OUTPUT_FILE = "tokenized_song.txt"

# ====== FIND ONE MIDI FILE ======
midi_files = sorted(
    list(Path(DATA_DIR).glob("**/*.mid")) +
    list(Path(DATA_DIR).glob("**/*.midi"))
)

if not midi_files:
    raise FileNotFoundError(f"No MIDI files found in {DATA_DIR}")

midi_path = midi_files[0]

print("Using MIDI file:")
print(midi_path)
print()

# ====== CREATE TOKENIZER ======
config = TokenizerConfig(
    pitch_range=(21, 109),      # piano range
    beat_res={(0, 4): 8},       # 8 positions per beat
    num_velocities=16,
    use_chords=False,
    use_rests=True,
    use_tempos=True,
    use_time_signatures=False,
)

tokenizer = REMI(config)

# ====== TOKENIZE FULL SONG ======
tok_sequences = tokenizer(midi_path)

print("Type returned by tokenizer:")
print(type(tok_sequences))
print()

print("Vocabulary size:")
print(len(tokenizer.vocab))
print()

# ====== HANDLE SINGLE SEQUENCE OR LIST OF SEQUENCES ======
if isinstance(tok_sequences, list):
    print("Number of token sequences / tracks:")
    print(len(tok_sequences))
    print()

    for track_idx, seq in enumerate(tok_sequences):
        print(f"=== Track {track_idx} ===")
        print("Number of tokens:")
        print(len(seq.ids))
        print()

        print(f"First {MAX_PRINT} token IDs:")
        print(seq.ids[:MAX_PRINT])
        print()

        print(f"First {MAX_PRINT} readable tokens:")
        print(seq.tokens[:MAX_PRINT])
        print()

        print("Readable tokens, one per line:")
        for i, token in enumerate(seq.tokens[:MAX_PRINT]):
            print(f"{i:04d}: {token}")

        print()
else:
    seq = tok_sequences

    print("Number of tokens:")
    print(len(seq.ids))
    print()

    print(f"First {MAX_PRINT} token IDs:")
    print(seq.ids[:MAX_PRINT])
    print()

    print(f"First {MAX_PRINT} readable tokens:")
    print(seq.tokens[:MAX_PRINT])
    print()

    print("Readable tokens, one per line:")
    for i, token in enumerate(seq.tokens[:MAX_PRINT]):
        print(f"{i:04d}: {token}")

# ====== SAVE ALL READABLE TOKENS TO FILE ======
with open(OUTPUT_FILE, "w") as f:
    f.write(f"MIDI file: {midi_path}\n\n")

    if isinstance(tok_sequences, list):
        for track_idx, seq in enumerate(tok_sequences):
            f.write(f"=== Track {track_idx} ===\n")
            f.write(f"Number of tokens: {len(seq.ids)}\n\n")

            for i, token in enumerate(seq.tokens):
                f.write(f"{i:04d}: {token}\n")

            f.write("\n")
    else:
        seq = tok_sequences
        f.write(f"Number of tokens: {len(seq.ids)}\n\n")

        for i, token in enumerate(seq.tokens):
            f.write(f"{i:04d}: {token}\n")

print(f"Saved readable tokens to {OUTPUT_FILE}")