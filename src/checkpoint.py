# src/checkpoint.py

import torch

from src.transformer import MusicTransformer


def save_checkpoint(
    path,
    model,
    optimizer,
    vocab_size,
    seq_len,
    d_model,
    nhead,
    num_layers,
    ff_dim,
    dropout,
    pad_id,
    bos_id,
    eos_id,
    mask_id,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "ff_dim": ff_dim,
        "dropout": dropout,
        "pad_id": pad_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "mask_id": mask_id,
    }

    torch.save(checkpoint, path)


def load_model_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = MusicTransformer(
        vocab_size=checkpoint["vocab_size"],
        d_model=checkpoint["d_model"],
        nhead=checkpoint["nhead"],
        num_layers=checkpoint["num_layers"],
        dim_feedforward=checkpoint["ff_dim"],
        dropout=checkpoint["dropout"],
        max_len=checkpoint["seq_len"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint