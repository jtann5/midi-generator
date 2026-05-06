# generate_random.py

from pathlib import Path

from src.device import get_device
from src.config import load_config
from src.checkpoint import load_model_from_checkpoint
from src.tokenizer_profiles import load_tokenizer_from_model_dir
from src.generation import (
    generate_tokens,
    remove_special_tokens,
    decode_ids_to_midi,
)


from pathlib import Path


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

        # Accept model folders with either final checkpoint or epoch checkpoints.
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
    """
    Sort epoch checkpoints numerically, then put checkpoint.pt at the end.
    """
    name = path.name

    if name == "checkpoint.pt":
        return (1, 999999)

    # checkpoint_epoch_005.pt -> 5
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


def main():
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

    print("\nOverride generation settings, or press Enter to keep defaults.\n")

    generate_length = ask_optional_int("Generate length", generate_length)
    temperature = ask_optional_float("Temperature", temperature)
    top_k = ask_optional_top_k(top_k)
    top_p = ask_optional_top_p(top_p)

    output_dir = Path("samples") / model_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = input(
        f"Output MIDI filename [random_{model_dir.name}.mid]: "
    ).strip()

    if output_name == "":
        output_name = f"random_{model_dir.name}.mid"

    output_path = output_dir / output_name

    pad_id = checkpoint["pad_id"]
    bos_id = checkpoint["bos_id"]
    eos_id = checkpoint["eos_id"]
    mask_id = checkpoint["mask_id"]
    seq_len = checkpoint["seq_len"]

    print("\nGenerating randomly from BOS token...")

    generated_ids = generate_tokens(
        model=model,
        start_ids=[bos_id],
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

    print("Generated token count including special tokens:", len(generated_ids))
    print("Generated music token count:", len(music_ids))

    decode_ids_to_midi(
        tokenizer=tokenizer,
        ids=music_ids,
        output_path=output_path,
    )

    print(f"\nSaved generated MIDI to {output_path}")


if __name__ == "__main__":
    main()