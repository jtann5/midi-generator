# src/dataset.py

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def find_midi_files(data_dir, max_files=None):
    midi_files = sorted(
        list(Path(data_dir).glob("**/*.mid")) +
        list(Path(data_dir).glob("**/*.midi"))
    )

    if len(midi_files) == 0:
        raise FileNotFoundError(f"No MIDI files found in {data_dir}")

    if max_files is not None:
        midi_files = midi_files[:max_files]

    return midi_files


def tokenize_midi_files(
    tokenizer,
    midi_files,
    seq_len,
    bos_id,
    eos_id,
):
    encoded_sequences = []

    for midi_path in midi_files:
        try:
            tok_sequences = tokenizer(midi_path)

            if not isinstance(tok_sequences, list):
                tok_sequences = [tok_sequences]

            for seq in tok_sequences:
                ids = seq.ids

                if ids is None or len(ids) == 0:
                    continue

                ids = [bos_id] + ids + [eos_id]

                if len(ids) > seq_len + 1:
                    encoded_sequences.append(ids)

        except Exception as e:
            print(f"Could not tokenize {midi_path}: {e}")

    if len(encoded_sequences) == 0:
        raise RuntimeError(
            "No usable token sequences found. Try lowering SEQ_LEN or checking DATA_DIR."
        )

    return encoded_sequences


def build_training_chunks(encoded_sequences, seq_len, stride=None):
    if stride is None:
        stride = seq_len // 2

    X = []
    Y = []

    for seq in encoded_sequences:
        for i in range(0, len(seq) - seq_len, stride):
            x_chunk = seq[i:i + seq_len]
            y_chunk = seq[i + 1:i + seq_len + 1]

            X.append(x_chunk)
            Y.append(y_chunk)

    X = np.array(X, dtype=np.int32)
    Y = np.array(Y, dtype=np.int32)

    return X, Y


class MusicDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]