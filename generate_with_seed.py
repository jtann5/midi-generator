# generate_with_seed.py

from src.generation_cli import (
    load_generation_context,
    ask_generation_settings,
    ask_user_for_seed_midi,
    get_output_paths,
    run_generation_and_save,
    save_generation_log,
)
from src.tokenizer_utils import get_seed_ids_from_midi


def main():
    ctx = load_generation_context()

    checkpoint = ctx["checkpoint"]

    dataset_path = ctx["config"]["dataset"]["path"]
    seed_midi_path = ask_user_for_seed_midi(dataset_path)

    default_seed_tokens = min(128, checkpoint["seq_len"])

    settings = ask_generation_settings(
        config=ctx["config"],
        include_seed_tokens=True,
        default_seed_tokens=default_seed_tokens,
    )

    pad_id = checkpoint["pad_id"]
    bos_id = checkpoint["bos_id"]
    eos_id = checkpoint["eos_id"]
    mask_id = checkpoint["mask_id"]

    print("\nUsing seed MIDI:")
    print(seed_midi_path)

    seed_ids = get_seed_ids_from_midi(
        tokenizer=ctx["tokenizer"],
        midi_path=seed_midi_path,
        seed_tokens=settings["seed_tokens"],
        bos_id=bos_id,
        pad_id=pad_id,
        eos_id=eos_id,
        mask_id=mask_id,
    )

    print("Seed token count:", len(seed_ids))
    print("Generating from seed...")

    output_path, log_path = get_output_paths(
        model_dir=ctx["model_dir"],
        mode="seeded",
    )

    counts = run_generation_and_save(
        model=ctx["model"],
        tokenizer=ctx["tokenizer"],
        checkpoint=checkpoint,
        device=ctx["device"],
        start_ids=seed_ids,
        output_path=output_path,
        generate_length=settings["generate_length"],
        temperature=settings["temperature"],
        top_k=settings["top_k"],
        top_p=settings["top_p"],
        repetition_penalty=settings["repetition_penalty"],
        repetition_window=settings["repetition_window"],
    )

    save_generation_log(
        log_path=log_path,
        mode="seeded",
        model_dir=ctx["model_dir"],
        checkpoint_path=ctx["checkpoint_path"],
        config=ctx["config"],
        tokenizer_profile=ctx["tokenizer_profile"],
        settings=settings,
        seed_midi_path=seed_midi_path,
        seed_tokens=settings["seed_tokens"],
        output_path=output_path,
        generated_token_count=counts["generated_token_count"],
        music_token_count=counts["music_token_count"],
    )


if __name__ == "__main__":
    main()