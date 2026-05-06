# src/model_profiles.py

MODEL_PROFILES = {
    "tiny_seq128": {
        "seq_len": 128,
        "batch_size": 32,
        "epochs": 10,
        "learning_rate": 3e-4,
        "d_model": 128,
        "nhead": 4,
        "num_layers": 4,
        "ff_dim": 512,
        "dropout": 0.1,
    },

    "small_seq256": {
        "seq_len": 256,
        "batch_size": 16,
        "epochs": 20,
        "learning_rate": 3e-4,
        "d_model": 256,
        "nhead": 8,
        "num_layers": 6,
        "ff_dim": 1024,
        "dropout": 0.1,
    },

    "medium_seq256": {
        "seq_len": 256,
        "batch_size": 8,
        "epochs": 20,
        "learning_rate": 3e-4,
        "d_model": 384,
        "nhead": 8,
        "num_layers": 8,
        "ff_dim": 1536,
        "dropout": 0.1,
    },

    "small_seq512": {
        "seq_len": 512,
        "batch_size": 8,
        "epochs": 10,
        "learning_rate": 3e-4,
        "d_model": 256,
        "nhead": 8,
        "num_layers": 6,
        "ff_dim": 1024,
        "dropout": 0.1,
    },
}


GENERATION_PROFILES = {
    "stable": {
        "generate_length": 800,
        "temperature": 0.85,
        "top_k": 50,
        "top_p": None,
        "repetition_penalty": 1.05,
        "repetition_window": 128,
    },

    "balanced": {
        "generate_length": 800,
        "temperature": 0.9,
        "top_k": None,
        "top_p": 0.92,
        "repetition_penalty": 1.08,
        "repetition_window": 128,
    },

    "varied": {
        "generate_length": 1000,
        "temperature": 1.0,
        "top_k": None,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
        "repetition_window": 128,
    },
}


def get_model_profile(profile_name):
    if profile_name not in MODEL_PROFILES:
        valid = ", ".join(MODEL_PROFILES.keys())
        raise ValueError(f"Unknown model profile '{profile_name}'. Valid: {valid}")

    return dict(MODEL_PROFILES[profile_name])


def get_generation_profile(profile_name):
    if profile_name not in GENERATION_PROFILES:
        valid = ", ".join(GENERATION_PROFILES.keys())
        raise ValueError(f"Unknown generation profile '{profile_name}'. Valid: {valid}")

    return dict(GENERATION_PROFILES[profile_name])