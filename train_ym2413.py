# train_ym2413.py

from src.training_runner import run_training


if __name__ == "__main__":
    run_training(
        data_dir="datasets/YM2413-MDB-v1.0.0/midi/adjust_tempo_remove_delayed_inst",
        dataset_name="YM2413-MDB v1.0.0",
        tokenizer_profile="multi_instrument_remi",
        default_model_prefix="ym2413_multi_remi",
        notes="Multi-instrument REMI+ style transformer trained on YM2413-MDB.",
    )