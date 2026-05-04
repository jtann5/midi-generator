import os
import glob
import math
import random
import numpy as np
import pretty_midi
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(
    "models/music_transformer_checkpoint.pth",
    map_location=DEVICE,
    weights_only=False
)

model = TinyMusicTransformer(
    vocab_size=checkpoint["vocab_size"],
    d_model=checkpoint["d_model"],
    nhead=checkpoint["nhead"],
    num_layers=checkpoint["num_layers"],
    dim_feedforward=checkpoint["ff_dim"],
    dropout=checkpoint["dropout"],
    max_len=checkpoint["seq_len"]
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

token_to_idx = checkpoint["token_to_idx"]
idx_to_token = checkpoint["idx_to_token"]
SEQ_LEN = checkpoint["seq_len"]

print("Checkpoint loaded.")