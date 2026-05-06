# src/config.py

import json
from pathlib import Path
from datetime import datetime


def prepare_model_dir(model_dir, overwrite=False):
    """
    Create a model directory safely.

    If the directory already exists and contains files, refuse to continue
    unless overwrite=True.
    """
    model_dir = Path(model_dir)

    if model_dir.exists() and any(model_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"\nModel directory already exists and is not empty:\n"
                f"  {model_dir}\n\n"
                f"Refusing to overwrite existing model files.\n"
                f"Choose a new MODEL_NAME or set OVERWRITE_MODEL_DIR = True."
            )

    model_dir.mkdir(parents=True, exist_ok=True)

    return model_dir


def make_experiment_config(
    *,
    model_name,
    dataset_name,
    dataset_path,
    tokenizer_profile,
    checkpoint_filename="checkpoint.pt",

    model_profile_name=None,
    generation_profile_name=None,

    seq_len,
    chunk_stride=None,
    batch_size,
    epochs,
    learning_rate,

    d_model,
    nhead,
    num_layers,
    ff_dim,
    dropout,

    max_files=None,

    generate_length=None,
    temperature=None,
    top_k=None,
    top_p=None,
    repetition_penalty=None,
    repetition_window=None,

    notes=None,
):
    return {
        "model_name": model_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),

        "profiles": {
            "model_profile": model_profile_name,
            "generation_profile": generation_profile_name,
        },

        "dataset": {
            "name": dataset_name,
            "path": dataset_path,
            "max_files": max_files,
        },

        "tokenizer": {
            "profile": tokenizer_profile,
            "profile_file": "tokenizer_profile.json",
        },

        "checkpoint": {
            "filename": checkpoint_filename,
        },

        "training": {
            "seq_len": seq_len,
            "chunk_stride": chunk_stride,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
        },

        "model": {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "ff_dim": ff_dim,
            "dropout": dropout,
        },

        "generation_defaults": {
            "generate_length": generate_length,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "repetition_window": repetition_window,
        },

        "notes": notes or "",
    }


def save_config(config, model_dir):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    config_path = model_dir / "config.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return config_path


def load_config(model_dir):
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found in {model_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def update_config(model_dir, updates):
    config = load_config(model_dir)

    for key, value in updates.items():
        config[key] = value

    save_config(config, model_dir)

    return config