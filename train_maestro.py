# train_maestro.py

from src.training_runner import run_training


if __name__ == "__main__":
    run_training(
        data_dir="datasets/maestro-v3.0.0",
        dataset_name="MAESTRO v3.0.0",
        tokenizer_profile="piano_remi",
        default_model_prefix="maestro_piano_remi",
        notes="Piano-specific REMI transformer trained on MAESTRO.",
    )