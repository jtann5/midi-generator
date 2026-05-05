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


def list_model_dirs(models_root="models"):
    models_root = Path(models_root)

    if not models_root.exists():
        return []

    model_dirs = []

    for path in sorted(models_root.iterdir()):
        if not path.is_dir():
            continue

        has_checkpoint = (path / "checkpoint.pt").exists()
        has_config = (path / "config.json").exists()
        has_tokenizer = (path / "tokenizer_profile.json").exists()

        if has_checkpoint and has_config and has_tokenizer:
            model_dirs.append(path)

    return model_dirs


def ask_user_for_model_dir():
    model_dirs = list_model_dirs("models")

    if len(model_dirs) == 0:
        raise FileNotFoundError(
            "No complete model folders found in models/. "
            "Expected checkpoint.pt, config.json, and tokenizer_profile.json."
        )

    print("\nAvailable models:\n")

    for i, model_dir in enumerate(model_dirs, start=1):
        print(f"{i}. {model_dir}")

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

    checkpoint_path = model_dir / config["checkpoint"]["filename"]

    model, checkpoint = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    generation_defaults = config.get("generation_defaults", {})

    generate_length = generation_defaults.get("generate_length", 800)
    temperature = generation_defaults.get("temperature", 0.85)
    top_k = generation_defaults.get("top_k", 50)
    top_p = generation_defaults.get("top_p", None)

    print("\nGeneration defaults from config:")
    print("generate_length:", generate_length)
    print("temperature:", temperature)
    print("top_k:", top_k)
    print("top_p:", top_p)

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