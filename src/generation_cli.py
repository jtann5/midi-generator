# src/generation_cli.py

from pathlib import Path
import random
import json
from datetime import datetime

from src.device import get_device
from src.config import load_config
from src.checkpoint import load_model_from_checkpoint
from src.tokenizer_profiles import load_tokenizer_from_model_dir
from src.tokenizer_utils import get_seed_ids_from_midi
from src.generation import (
    generate_tokens,
    remove_special_tokens,
    decode_ids_to_midi,
)


def list_model_dirs(models_root="models"):
    models_root = Path(models_root)

    if not models_root.exists():
        return []

    model_dirs = []

    for path in sorted(models_root.iterdir()):
        if not path.is_dir():
            continue

        has_config = (path / "config.json").exists()
        has_tokenizer = (path / "tokenizer_profile.json").exists()

        has_final_checkpoint = (path / "checkpoint.pt").exists()
        has_epoch_checkpoints = len(list(path.glob("checkpoint_epoch_*.pt"))) > 0

        if has_config and has_tokenizer and (has_final_checkpoint or has_epoch_checkpoints):
            model_dirs.append(path)

    return model_dirs


def ask_user_for_model_dir(models_root="models"):
    model_dirs = list_model_dirs(models_root)

    if len(model_dirs) == 0:
        raise FileNotFoundError(
            f"No complete model folders found in {models_root}/. "
            "Expected config.json, tokenizer_profile.json, and at least one checkpoint."
        )

    print("\nAvailable models:\n")

    for i, model_dir in enumerate(model_dirs, start=1):
        print(f"{i}. {model_dir.name}")

    print()

    while True:
        choice = input("Which model do you want to use? Enter number: ").strip()

        try:
            index = int(choice) - 1

            if 0 <= index < len(model_dirs):
                return model_dirs[index]

        except ValueError:
            pass

        print("Invalid choice. Try again.")


def checkpoint_sort_key(path):
    name = path.name

    if name == "checkpoint.pt":
        return (1, 999999)

    if name.startswith("checkpoint_epoch_") and name.endswith(".pt"):
        epoch_text = name.replace("checkpoint_epoch_", "").replace(".pt", "")

        try:
            return (0, int(epoch_text))
        except ValueError:
            return (0, 999999)

    return (0, 999999)


def list_checkpoints(model_dir):
    model_dir = Path(model_dir)

    checkpoint_paths = list(model_dir.glob("checkpoint_epoch_*.pt"))

    final_checkpoint = model_dir / "checkpoint.pt"
    if final_checkpoint.exists():
        checkpoint_paths.append(final_checkpoint)

    checkpoint_paths = sorted(checkpoint_paths, key=checkpoint_sort_key)

    if len(checkpoint_paths) == 0:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")

    return checkpoint_paths


def ask_user_for_checkpoint(model_dir):
    checkpoint_paths = list_checkpoints(model_dir)

    print(f"\nAvailable checkpoints for {Path(model_dir).name}:\n")

    for i, checkpoint_path in enumerate(checkpoint_paths, start=1):
        label = checkpoint_path.name

        if checkpoint_path.name == "checkpoint.pt":
            label += "  ← final/latest"

        print(f"{i}. {label}")

    print()

    while True:
        choice = input("Which checkpoint do you want to use? Enter number: ").strip()

        try:
            index = int(choice) - 1

            if 0 <= index < len(checkpoint_paths):
                return checkpoint_paths[index]

        except ValueError:
            pass

        print("Invalid choice. Try again.")


def ask_optional_int(prompt, default):
    value = input(f"{prompt} [{default}]: ").strip()

    if value == "":
        return default

    return int(value)


def ask_optional_float(prompt, default):
    value = input(f"{prompt} [{default}]: ").strip()

    if value == "":
        return default

    return float(value)


def ask_optional_top_k(default):
    value = input(f"top_k [{default}, blank keeps default, 0 disables]: ").strip()

    if value == "":
        return default

    value = int(value)

    if value <= 0:
        return None

    return value


def ask_optional_top_p(default):
    value = input(f"top_p [{default}, blank keeps default, 0 disables]: ").strip()

    if value == "":
        return default

    value = float(value)

    if value <= 0:
        return None

    return value


def ask_optional_repetition_penalty(default):
    value = input(f"repetition_penalty [{default}, blank keeps default, 1 disables]: ").strip()

    if value == "":
        return default

    value = float(value)

    if value <= 1.0:
        return None

    return value


def ask_optional_repetition_window(default):
    value = input(f"repetition_window [{default}]: ").strip()

    if value == "":
        return default

    return int(value)


def load_generation_context():
    model_dir = ask_user_for_model_dir()

    config = load_config(model_dir)
    tokenizer, tokenizer_profile = load_tokenizer_from_model_dir(model_dir)

    device = get_device()
    print("Using device:", device)

    checkpoint_path = ask_user_for_checkpoint(model_dir)

    model, checkpoint = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    print(f"Loaded model: {model_dir.name}")
    print(f"Loaded checkpoint: {checkpoint_path.name}")

    return {
        "model_dir": model_dir,
        "config": config,
        "tokenizer": tokenizer,
        "tokenizer_profile": tokenizer_profile,
        "device": device,
        "checkpoint_path": checkpoint_path,
        "model": model,
        "checkpoint": checkpoint,
    }


def ask_generation_settings(config, include_seed_tokens=False, default_seed_tokens=None):
    generation_defaults = config.get("generation_defaults", {})

    generate_length = generation_defaults.get("generate_length", 800)
    temperature = generation_defaults.get("temperature", 0.85)
    top_k = generation_defaults.get("top_k", 50)
    top_p = generation_defaults.get("top_p", None)
    repetition_penalty = generation_defaults.get("repetition_penalty", 1.08)
    repetition_window = generation_defaults.get("repetition_window", 128)

    print("\nGeneration defaults from config:")
    print("generate_length:", generate_length)
    print("temperature:", temperature)
    print("top_k:", top_k)
    print("top_p:", top_p)
    print("repetition_penalty:", repetition_penalty)
    print("repetition_window:", repetition_window)

    if include_seed_tokens:
        print("default seed tokens:", default_seed_tokens)

    print("\nOverride generation settings, or press Enter to keep defaults.\n")

    settings = {}

    if include_seed_tokens:
        settings["seed_tokens"] = ask_optional_int("Seed tokens", default_seed_tokens)

    settings["generate_length"] = ask_optional_int("Generate length", generate_length)
    settings["temperature"] = ask_optional_float("Temperature", temperature)
    settings["top_k"] = ask_optional_top_k(top_k)
    settings["top_p"] = ask_optional_top_p(top_p)
    settings["repetition_penalty"] = ask_optional_repetition_penalty(repetition_penalty)
    settings["repetition_window"] = ask_optional_repetition_window(repetition_window)

    return settings


def find_midi_files(data_dir):
    data_dir = Path(data_dir)

    return sorted(
        list(data_dir.glob("**/*.mid")) +
        list(data_dir.glob("**/*.midi"))
    )


def ask_user_for_seed_midi(default_data_dir):
    print("\nSeed MIDI selection:")
    print("Enter a MIDI path, or press Enter to randomly choose one from the model dataset.")

    seed_path_text = input("Seed MIDI path: ").strip()

    if seed_path_text != "":
        seed_path = Path(seed_path_text)

        if not seed_path.exists():
            raise FileNotFoundError(f"Seed MIDI does not exist: {seed_path}")

        return seed_path

    midi_files = find_midi_files(default_data_dir)

    if len(midi_files) == 0:
        raise FileNotFoundError(
            f"No MIDI files found in saved dataset path: {default_data_dir}"
        )

    seed_path = random.choice(midi_files)

    print(f"Randomly selected seed MIDI: {seed_path}")

    return seed_path


def run_generation_and_save(
    *,
    model,
    tokenizer,
    checkpoint,
    device,
    start_ids,
    output_path,
    generate_length,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    repetition_window,
):
    pad_id = checkpoint["pad_id"]
    bos_id = checkpoint["bos_id"]
    eos_id = checkpoint["eos_id"]
    mask_id = checkpoint["mask_id"]
    seq_len = checkpoint["seq_len"]

    generated_ids = generate_tokens(
        model=model,
        start_ids=start_ids,
        max_new_tokens=generate_length,
        seq_len=seq_len,
        device=device,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        mask_id=mask_id,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        repetition_window=repetition_window,
    )

    music_ids = remove_special_tokens(
        generated_ids,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        mask_id=mask_id,
    )

    generated_token_count = len(generated_ids)
    music_token_count = len(music_ids)

    print("Generated token count including special tokens:", generated_token_count)
    print("Generated music token count:", music_token_count)

    decode_ids_to_midi(
        tokenizer=tokenizer,
        ids=music_ids,
        output_path=output_path,
    )

    print(f"\nSaved generated MIDI to {output_path}")

    return {
        "generated_token_count": generated_token_count,
        "music_token_count": music_token_count,
    }


def get_next_sample_index(output_dir, mode):
    """
    Find the next available sample number for a mode.

    Example:
        random_001.mid
        random_002.mid
        seeded_001.mid
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = list(output_dir.glob(f"{mode}_*.mid"))

    max_index = 0

    for path in existing:
        stem = path.stem  # random_001
        parts = stem.split("_")

        if len(parts) < 2:
            continue

        index_text = parts[-1]

        if index_text.isdigit():
            max_index = max(max_index, int(index_text))

    return max_index + 1


def get_output_paths(model_dir, mode):
    """
    Ask for output filename, with an incrementing default.

    Example defaults:
        samples/model_name/random_001.mid
        samples/model_name/seeded_001.mid
    """
    output_dir = Path("samples") / Path(model_dir).name
    output_dir.mkdir(parents=True, exist_ok=True)

    index = get_next_sample_index(output_dir, mode)
    default_base_name = f"{mode}_{index:03d}"
    default_midi_name = f"{default_base_name}.mid"

    output_name = input(f"Output MIDI filename [{default_midi_name}]: ").strip()

    if output_name == "":
        output_name = default_midi_name

    if not output_name.lower().endswith((".mid", ".midi")):
        output_name += ".mid"

    midi_path = output_dir / output_name
    log_path = midi_path.with_suffix(".txt")

    return midi_path, log_path

def make_json_safe(obj):
    """
    Recursively convert objects into JSON-safe structures.

    Fixes cases like:
        {(0, 4): 8}

    by converting tuple keys to strings.
    """
    if isinstance(obj, dict):
        safe_dict = {}

        for key, value in obj.items():
            if isinstance(key, tuple):
                safe_key = str(key)
            else:
                safe_key = key

            safe_dict[safe_key] = make_json_safe(value)

        return safe_dict

    if isinstance(obj, list):
        return [make_json_safe(item) for item in obj]

    if isinstance(obj, tuple):
        return list(obj)

    return obj

def save_generation_log(
    *,
    log_path,
    mode,
    model_dir,
    checkpoint_path,
    config,
    tokenizer_profile,
    settings,
    seed_midi_path=None,
    seed_tokens=None,
    output_path=None,
    generated_token_count=None,
    music_token_count=None,
):
    """
    Save a human-readable generation log next to the MIDI file.
    """
    log = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,

        "model": {
            "model_dir": str(model_dir),
            "model_name": Path(model_dir).name,
            "checkpoint": str(checkpoint_path),
            "checkpoint_name": Path(checkpoint_path).name,
        },

        "dataset": config.get("dataset", {}),
        "profiles": config.get("profiles", {}),
        "training": config.get("training", {}),
        "model_architecture": config.get("model", {}),
        "parameters": config.get("parameters", {}),
        "tokenizer_profile": tokenizer_profile,

        "generation_settings": settings,

        "seed": {
            "seed_midi_path": str(seed_midi_path) if seed_midi_path is not None else None,
            "seed_tokens": seed_tokens,
        },

        "output": {
            "midi_path": str(output_path) if output_path is not None else None,
            "generated_token_count": generated_token_count,
            "music_token_count": music_token_count,
        },
    }

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Generation Log\n")
        f.write("====================\n\n")

        f.write(f"Created at: {log['created_at']}\n")
        f.write(f"Mode: {mode}\n\n")

        f.write("Model\n")
        f.write("--------------------\n")
        f.write(f"Model name: {log['model']['model_name']}\n")
        f.write(f"Model dir: {log['model']['model_dir']}\n")
        f.write(f"Checkpoint: {log['model']['checkpoint_name']}\n\n")

        f.write("Generation settings\n")
        f.write("--------------------\n")
        for key, value in settings.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        if seed_midi_path is not None:
            f.write("Seed\n")
            f.write("--------------------\n")
            f.write(f"Seed MIDI: {seed_midi_path}\n")
            f.write(f"Seed tokens: {seed_tokens}\n\n")

        f.write("Training config\n")
        f.write("--------------------\n")
        f.write(json.dumps(make_json_safe({
            "dataset": log["dataset"],
            "profiles": log["profiles"],
            "training": log["training"],
            "model_architecture": log["model_architecture"],
            "parameters": log["parameters"],
        }), indent=2))
        f.write("\n\n")

        f.write("Tokenizer profile\n")
        f.write("--------------------\n")
        f.write(json.dumps(make_json_safe(tokenizer_profile), indent=2))
        f.write("\n\n")

        f.write("Output\n")
        f.write("--------------------\n")
        f.write(f"MIDI path: {output_path}\n")
        f.write(f"Generated token count: {generated_token_count}\n")
        f.write(f"Music token count: {music_token_count}\n")

    print(f"Saved generation log to {log_path}")