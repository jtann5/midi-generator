# src/tokenizer_utils.py


def get_token_id(tokenizer, token_name):
    """
    Safely get a token ID from MidiTok.
    MidiTok versions differ slightly, so this handles common cases.
    """
    if hasattr(tokenizer, "vocab") and token_name in tokenizer.vocab:
        return tokenizer.vocab[token_name]

    try:
        return tokenizer[token_name]
    except Exception as e:
        raise KeyError(f"Could not find token {token_name} in tokenizer vocab.") from e


def get_special_token_ids(tokenizer):
    return {
        "pad_id": get_token_id(tokenizer, "PAD_None"),
        "bos_id": get_token_id(tokenizer, "BOS_None"),
        "eos_id": get_token_id(tokenizer, "EOS_None"),
        "mask_id": get_token_id(tokenizer, "MASK_None"),
    }


def get_seed_ids_from_midi(
    tokenizer,
    midi_path,
    seed_tokens,
    bos_id,
    pad_id,
    eos_id,
    mask_id,
):
    tok_sequences = tokenizer(midi_path)

    if not isinstance(tok_sequences, list):
        tok_sequences = [tok_sequences]

    seq = max(tok_sequences, key=lambda s: len(s.ids) if s.ids is not None else 0)

    ids = seq.ids

    if ids is None or len(ids) == 0:
        raise ValueError(f"Seed MIDI produced no tokens: {midi_path}")

    ids = [
        token_id for token_id in ids
        if token_id not in {pad_id, bos_id, eos_id, mask_id}
    ]

    return [bos_id] + ids[:seed_tokens]