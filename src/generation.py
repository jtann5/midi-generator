# src/generation.py

import torch
from miditok import TokSequence


def generate_tokens(
    model,
    start_ids,
    max_new_tokens,
    seq_len,
    device,
    pad_id,
    bos_id,
    eos_id,
    mask_id,
    temperature=0.8,
    top_k=50,
    top_p=None,
    repetition_penalty=1.08,
    repetition_window=128,
    print_every=100,
):
    model.eval()

    generated = list(start_ids)
    forbidden_ids = {pad_id, bos_id, mask_id}

    with torch.no_grad():
        for step in range(max_new_tokens):
            input_ids = generated[-seq_len:]
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

            logits = model(input_tensor)
            next_logits = logits[:, -1, :] / temperature

            # Avoid non-musical special tokens.
            for bad_id in forbidden_ids:
                next_logits[:, bad_id] = -float("inf")

            # Penalize recently repeated tokens.
            next_logits = apply_repetition_penalty(
                next_logits,
                generated_ids=generated,
                penalty=repetition_penalty,
                window=repetition_window,
            )

            # Apply top-k / top-p filtering if you added it.
            next_logits = apply_top_k_top_p_filtering(
                next_logits,
                top_k=top_k,
                top_p=top_p,
            )

            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_id)

            if print_every is not None and (step + 1) % print_every == 0:
                print(f"Generated {step + 1}/{max_new_tokens} tokens...")

            if next_id == eos_id:
                print("Generated EOS. Stopping.")
                break

    return generated


def remove_special_tokens(ids, pad_id, bos_id, eos_id, mask_id):
    return [
        token_id for token_id in ids
        if token_id not in {pad_id, bos_id, eos_id, mask_id}
    ]


def decode_ids_to_midi(tokenizer, ids, output_path):
    tok_seq = TokSequence(ids=ids)

    try:
        tokenizer.complete_sequence(tok_seq)
    except Exception:
        pass

    decode_attempts = [
        lambda: tokenizer.decode(tok_seq),
        lambda: tokenizer.decode([tok_seq]),
        lambda: tokenizer(tok_seq),
    ]

    last_error = None

    for attempt in decode_attempts:
        try:
            midi_obj = attempt()

            if hasattr(midi_obj, "dump_midi"):
                midi_obj.dump_midi(output_path)
                return

            if hasattr(midi_obj, "write"):
                midi_obj.write(output_path)
                return

            if hasattr(midi_obj, "dump"):
                midi_obj.dump(output_path)
                return

            raise RuntimeError(f"Decoded object has no known save method: {type(midi_obj)}")

        except Exception as e:
            last_error = e

    raise RuntimeError(f"Could not decode generated tokens to MIDI: {last_error}")

def apply_top_k_top_p_filtering(logits, top_k=None, top_p=None):
    """
    Filter logits using top-k and/or nucleus top-p sampling.

    top_k:
        Keep only the k most likely tokens.

    top_p:
        Keep the smallest set of tokens whose cumulative probability
        is at least top_p.
    """
    logits = logits.clone()

    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))

        values, _ = torch.topk(logits, top_k)
        cutoff = values[:, -1].unsqueeze(-1)

        logits[logits < cutoff] = -float("inf")

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens after cumulative probability exceeds top_p.
        sorted_indices_to_remove = cumulative_probs > top_p

        # Keep at least the first token above the threshold.
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1,
            index=sorted_indices,
            src=sorted_indices_to_remove,
        )

        logits[indices_to_remove] = -float("inf")

    return logits

def apply_repetition_penalty(logits, generated_ids, penalty=1.08, window=128):
    """
    Penalize tokens that appeared recently.

    penalty > 1.0 makes repeated recent tokens less likely.
    window controls how far back we look.
    """
    if penalty is None or penalty <= 1.0:
        return logits

    recent_ids = generated_ids[-window:]

    for token_id in set(recent_ids):
        # HuggingFace-style repetition penalty logic:
        # If the logit is positive, divide it.
        # If the logit is negative, multiply it.
        if logits[0, token_id] > 0:
            logits[0, token_id] /= penalty
        else:
            logits[0, token_id] *= penalty

    return logits