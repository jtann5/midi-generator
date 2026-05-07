# generate_random.py

from src.generation_cli import (
    load_generation_context,
    ask_generation_settings,
    get_output_paths,
    run_generation_and_save,
    save_generation_log,
)


def main():
    ctx = load_generation_context()

    checkpoint = ctx["checkpoint"]
    bos_id = checkpoint["bos_id"]

    settings = ask_generation_settings(
        config=ctx["config"],
        include_seed_tokens=False,
    )

    output_path, log_path = get_output_paths(
        model_dir=ctx["model_dir"],
        mode="random",
    )

    print("\nGenerating randomly from BOS token...")

    counts = run_generation_and_save(
        model=ctx["model"],
        tokenizer=ctx["tokenizer"],
        checkpoint=checkpoint,
        device=ctx["device"],
        start_ids=[bos_id],
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
        mode="random",
        model_dir=ctx["model_dir"],
        checkpoint_path=ctx["checkpoint_path"],
        config=ctx["config"],
        tokenizer_profile=ctx["tokenizer_profile"],
        settings=settings,
        output_path=output_path,
        generated_token_count=counts["generated_token_count"],
        music_token_count=counts["music_token_count"],
    )


if __name__ == "__main__":
    main()