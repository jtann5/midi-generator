# src/tokenizer_profiles.py

import json
from pathlib import Path

from miditok import REMI, TokenizerConfig


SPECIAL_TOKENS = [
    "PAD_None",
    "BOS_None",
    "EOS_None",
    "MASK_None",
]


TOKENIZER_PROFILES = {
    "piano_remi": {
        "description": "Piano-specific REMI tokenizer for MAESTRO-style solo piano MIDI.",
        "tokenizer": "REMI",
        "pitch_range": (21, 109),  # standard piano range: A0-C8
        "beat_res": {(0, 4): 8},
        "num_velocities": 16,

        "use_chords": False,
        "use_rests": True,
        "use_drums": False,
        "use_tempos": True,
        "use_time_signatures": True,

        "use_programs": False,
        "one_token_stream_for_programs": False,

        "special_tokens": SPECIAL_TOKENS,
    },

    "multi_instrument_remi": {
        "description": "Multi-instrument REMI+ style tokenizer with program tokens.",
        "tokenizer": "REMI",
        "pitch_range": (0, 127),
        "beat_res": {(0, 4): 8},
        "num_velocities": 16,

        "use_chords": False,
        "use_rests": True,
        "use_drums": True,
        "use_tempos": True,
        "use_time_signatures": True,

        "use_programs": True,
        "one_token_stream_for_programs": True,

        "special_tokens": SPECIAL_TOKENS,
    },
}


def _json_safe_profile(profile):
    """
    Convert tuple/dict keys into JSON-safe forms.
    MidiTok wants beat_res like {(0, 4): 8}, but JSON cannot store tuple keys.
    """
    profile = dict(profile)

    profile["pitch_range"] = list(profile["pitch_range"])

    # Convert {(0, 4): 8} -> [{"start": 0, "end": 4, "resolution": 8}]
    beat_res = []
    for beat_range, resolution in profile["beat_res"].items():
        beat_res.append({
            "start": beat_range[0],
            "end": beat_range[1],
            "resolution": resolution,
        })

    profile["beat_res"] = beat_res

    return profile


def _profile_from_json_safe(profile):
    """
    Convert JSON-safe profile back into the format MidiTok expects.
    """
    profile = dict(profile)

    profile["pitch_range"] = tuple(profile["pitch_range"])

    beat_res = {}
    for item in profile["beat_res"]:
        beat_res[(item["start"], item["end"])] = item["resolution"]

    profile["beat_res"] = beat_res

    return profile


def get_tokenizer_profile(profile_name):
    if profile_name not in TOKENIZER_PROFILES:
        valid = ", ".join(TOKENIZER_PROFILES.keys())
        raise ValueError(f"Unknown tokenizer profile '{profile_name}'. Valid profiles: {valid}")

    return dict(TOKENIZER_PROFILES[profile_name])


def create_tokenizer_from_profile(profile_name):
    profile = get_tokenizer_profile(profile_name)

    config = TokenizerConfig(
        pitch_range=profile["pitch_range"],
        beat_res=profile["beat_res"],
        num_velocities=profile["num_velocities"],

        use_chords=profile["use_chords"],
        use_rests=profile["use_rests"],
        use_drums=profile["use_drums"],
        use_tempos=profile["use_tempos"],
        use_time_signatures=profile["use_time_signatures"],

        use_programs=profile["use_programs"],
        one_token_stream_for_programs=profile["one_token_stream_for_programs"],

        special_tokens=profile["special_tokens"],
    )

    return REMI(config)


def save_tokenizer_profile(profile_name, model_dir):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    profile = get_tokenizer_profile(profile_name)
    profile = _json_safe_profile(profile)

    output_path = model_dir / "tokenizer_profile.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    return output_path


def load_tokenizer_from_model_dir(model_dir):
    model_dir = Path(model_dir)
    profile_path = model_dir / "tokenizer_profile.json"

    if not profile_path.exists():
        raise FileNotFoundError(f"No tokenizer_profile.json found in {model_dir}")

    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    profile = _profile_from_json_safe(profile)

    config = TokenizerConfig(
        pitch_range=profile["pitch_range"],
        beat_res=profile["beat_res"],
        num_velocities=profile["num_velocities"],

        use_chords=profile["use_chords"],
        use_rests=profile["use_rests"],
        use_drums=profile["use_drums"],
        use_tempos=profile["use_tempos"],
        use_time_signatures=profile["use_time_signatures"],

        use_programs=profile["use_programs"],
        one_token_stream_for_programs=profile["one_token_stream_for_programs"],

        special_tokens=profile["special_tokens"],
    )

    return REMI(config), profile