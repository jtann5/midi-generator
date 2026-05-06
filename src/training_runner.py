# src/training_runner.py

import argparse
import os
import re

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.device import get_device
from src.dataset import (
    find_midi_files,
    tokenize_midi_files,
    build_training_chunks,
    MusicDataset,
)
from src.transformer import MusicTransformer
from src.checkpoint import save_checkpoint
from src.config import make_experiment_config, save_config, prepare_model_dir, update_config, load_config
from src.tokenizer_profiles import (
    create_tokenizer_from_profile,
    save_tokenizer_profile,
    load_tokenizer_from_model_dir,
)
from src.tokenizer_utils import get_special_token_ids
from src.model_profiles import get_model_profile, get_generation_profile


def infer_epoch_from_checkpoint_path(checkpoint_path):
    """
    Infer completed epoch number from checkpoint filename.

    Example:
        checkpoint_epoch_012.pt -> 12
        checkpoint.pt -> 0
    """
    filename = os.path.basename(checkpoint_path)

    match = re.search(r"checkpoint_epoch_(\d+)\.pt$", filename)

    if match is None:
        return 0

    return int(match.group(1))


def parse_training_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--model-profile", type=str, default=None)
    parser.add_argument("--generation-profile", type=str, default=None)

    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume-checkpoint", type=str, default=None)

    # Optional overrides. If left as None, profile values are used.
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--chunk-stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)

    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--ff-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)

    return parser.parse_args()


def validate_resume_args(args):
    if args.resume_checkpoint is None:
        return

    incompatible_flags = []

    disallowed = {
        "--model-name": args.model_name,
        "--model-profile": args.model_profile,
        "--generation-profile": args.generation_profile,
        "--max-files": args.max_files,
        "--overwrite": args.overwrite if args.overwrite else None,
        "--seq-len": args.seq_len,
        "--chunk-stride": args.chunk_stride,
        "--batch-size": args.batch_size,
        "--lr": args.lr,
        "--d-model": args.d_model,
        "--nhead": args.nhead,
        "--num-layers": args.num_layers,
        "--ff-dim": args.ff_dim,
        "--dropout": args.dropout,
    }

    for flag_name, value in disallowed.items():
        if value is not None:
            incompatible_flags.append(flag_name)

    if incompatible_flags:
        flags = ", ".join(incompatible_flags)
        raise ValueError(
            "In resume mode, the saved config.json is the source of truth.\n"
            f"Remove these incompatible flags: {flags}\n\n"
            "Use:\n"
            "  python train_maestro.py "
            "--resume-checkpoint models/<model_name>/checkpoint_epoch_012.pt\n\n"
            "Allowed override:\n"
            "  --epochs <new_total_epochs>"
        )


def run_training(
    *,
    data_dir,
    dataset_name,
    tokenizer_profile,
    default_model_prefix,
    notes,
):
    args = parse_training_args()

    is_resuming = args.resume_checkpoint is not None

    if is_resuming:
        validate_resume_args(args)
    else:
        # Default values
        if args.model_profile is None:
            args.model_profile = "small_seq256"

        if args.generation_profile is None:
            args.generation_profile = "balanced"

        if args.max_files is None:
            args.max_files = 500

    if is_resuming:
        model_profile_name = None
        generation_profile_name = None
        model_profile = None
        generation_profile = None
    else:
        model_profile_name = args.model_profile
        generation_profile_name = args.generation_profile

        model_profile = get_model_profile(model_profile_name)
        generation_profile = get_generation_profile(generation_profile_name)

    if args.resume_checkpoint is not None and args.model_name is None:
        model_dir = os.path.dirname(args.resume_checkpoint)
        model_name = os.path.basename(model_dir)
    else:
        model_name = (
            args.model_name
            if args.model_name is not None
            else f"{default_model_prefix}_{model_profile_name}_v1"
        )
        model_dir = f"models/{model_name}"

    checkpoint_filename = "checkpoint.pt"
    checkpoint_path = os.path.join(model_dir, checkpoint_filename)

    overwrite_model_dir = args.overwrite

    if is_resuming:
        saved_config = load_config(model_dir)
        max_files = saved_config["dataset"]["max_files"]
        seq_len = saved_config["training"]["seq_len"]
        chunk_stride = saved_config["training"]["chunk_stride"]
        batch_size = saved_config["training"]["batch_size"]
        epochs = saved_config["training"]["epochs"]
        lr = saved_config["training"]["learning_rate"]
        d_model = saved_config["model"]["d_model"]
        nhead = saved_config["model"]["nhead"]
        num_layers = saved_config["model"]["num_layers"]
        ff_dim = saved_config["model"]["ff_dim"]
        dropout = saved_config["model"]["dropout"]
        generate_length = saved_config["generation_defaults"]["generate_length"]
        temperature = saved_config["generation_defaults"]["temperature"]
        top_k = saved_config["generation_defaults"]["top_k"]
        top_p = saved_config["generation_defaults"]["top_p"]
        repetition_penalty = saved_config["generation_defaults"]["repetition_penalty"]
        repetition_window = saved_config["generation_defaults"]["repetition_window"]
    else:
        max_files = args.max_files

        seq_len = args.seq_len if args.seq_len is not None else model_profile["seq_len"]
        chunk_stride = args.chunk_stride if args.chunk_stride is not None else seq_len // 2
        batch_size = args.batch_size if args.batch_size is not None else model_profile["batch_size"]
        epochs = args.epochs if args.epochs is not None else model_profile["epochs"]
        lr = args.lr if args.lr is not None else model_profile["learning_rate"]

        d_model = args.d_model if args.d_model is not None else model_profile["d_model"]
        nhead = args.nhead if args.nhead is not None else model_profile["nhead"]
        num_layers = args.num_layers if args.num_layers is not None else model_profile["num_layers"]
        ff_dim = args.ff_dim if args.ff_dim is not None else model_profile["ff_dim"]
        dropout = args.dropout if args.dropout is not None else model_profile["dropout"]

        generate_length = generation_profile["generate_length"]
        temperature = generation_profile["temperature"]
        top_k = generation_profile["top_k"]
        top_p = generation_profile["top_p"]
        repetition_penalty = generation_profile["repetition_penalty"]
        repetition_window = generation_profile["repetition_window"]

    if is_resuming and args.epochs is not None:
        print(f"Overriding resume target epochs: {epochs} -> {args.epochs}")

        epochs = args.epochs

    print("\nResolved experiment settings:")
    print("Model name:", model_name)
    print("Model dir:", model_dir)
    print("Dataset:", dataset_name)
    print("Data dir:", data_dir)
    print("Tokenizer profile:", tokenizer_profile)
    print("Model profile:", model_profile_name)
    print("Generation profile:", generation_profile_name)
    print("Max files:", max_files)
    print("Overwrite model dir:", overwrite_model_dir)

    print("\nTraining/model settings:")
    print("Seq len:", seq_len)
    print("Chunk stride:", chunk_stride)
    print("Batch size:", batch_size)
    print("Epochs:", epochs)
    print("Learning rate:", lr)
    print("D_MODEL:", d_model)
    print("NHEAD:", nhead)
    print("NUM_LAYERS:", num_layers)
    print("FF_DIM:", ff_dim)
    print("DROPOUT:", dropout)

    print("\nGeneration defaults:")
    print("Generate length:", generate_length)
    print("Temperature:", temperature)
    print("Top k:", top_k)
    print("Top p:", top_p)
    print("Repetition penalty:", repetition_penalty)
    print("Repetition window:", repetition_window)
    print()

    # =========================
    # SETUP
    # =========================

    if is_resuming:
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory does not exist for resume: {model_dir}")

        if not os.path.exists(args.resume_checkpoint):
            raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume_checkpoint}")

        print("Resume mode: allowing existing model directory.")
    else:
        prepare_model_dir(model_dir, overwrite=overwrite_model_dir)

    device = get_device()
    print("Using device:", device)


    if is_resuming:
        tokenizer, _ = load_tokenizer_from_model_dir(model_dir)
        print("Resume mode: keeping existing config.json and tokenizer_profile.json.")
    else:
        tokenizer = create_tokenizer_from_profile(tokenizer_profile)
        save_tokenizer_profile(tokenizer_profile, model_dir)

        config = make_experiment_config(
            model_name=model_name,
            dataset_name=dataset_name,
            dataset_path=data_dir,
            tokenizer_profile=tokenizer_profile,
            checkpoint_filename=checkpoint_filename,

            # Add these to make_experiment_config if you haven't yet.
            model_profile_name=model_profile_name,
            generation_profile_name=generation_profile_name,

            seq_len=seq_len,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=lr,

            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,

            max_files=max_files,
            chunk_stride=chunk_stride,

            generate_length=generate_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,

            notes=notes,
        )

        save_config(config, model_dir)


    special_ids = get_special_token_ids(tokenizer)
    pad_id = special_ids["pad_id"]
    bos_id = special_ids["bos_id"]
    eos_id = special_ids["eos_id"]
    mask_id = special_ids["mask_id"]

    vocab_size = len(tokenizer.vocab)

    print("Vocab size:", vocab_size)
    print("PAD id:", pad_id)
    print("BOS id:", bos_id)
    print("EOS id:", eos_id)
    print("MASK id:", mask_id)

    # =========================
    # DATASET
    # =========================

    midi_files = find_midi_files(data_dir, max_files=max_files)

    print("Total files used:", len(midi_files))
    print("First file:", midi_files[0])

    encoded_sequences = tokenize_midi_files(
        tokenizer=tokenizer,
        midi_files=midi_files,
        seq_len=seq_len,
        bos_id=bos_id,
        eos_id=eos_id,
    )

    print("Usable token sequences:", len(encoded_sequences))

    X, Y = build_training_chunks(
        encoded_sequences,
        seq_len,
        stride=chunk_stride,
    )

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    dataset = MusicDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Dataset size:", len(dataset))

    # =========================
    # MODEL
    # =========================

    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=ff_dim,
        dropout=dropout,
        max_len=seq_len,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if not is_resuming:
        update_config(
            model_dir,
            {
                "parameters": {
                    "total": total_params,
                    "trainable": trainable_params,
                }
            },
        )

    # =========================
    # TRAINING
    # =========================

    start_epoch = 0

    if args.resume_checkpoint is not None:
        print(f"Resuming from checkpoint: {args.resume_checkpoint}")

        resume_checkpoint = torch.load(args.resume_checkpoint, map_location=device)

        model.load_state_dict(resume_checkpoint["model_state_dict"])

        if resume_checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])

        start_epoch = infer_epoch_from_checkpoint_path(args.resume_checkpoint)

        print(f"Resumed from epoch {start_epoch}")
        print(f"Next epoch will be {start_epoch + 1}")

    for epoch in range(start_epoch, epochs):
        model.train()

        total_loss = 0.0
        num_batches = len(loader)

        for batch_idx, (xb, yb) in enumerate(loader, start=1):
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)

            loss = criterion(
                logits.reshape(-1, vocab_size),
                yb.reshape(-1),
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

            percent = 100 * batch_idx / num_batches

            print(
                f"\rEpoch {epoch + 1}/{epochs} - {percent:6.2f}% "
                f"({batch_idx}/{num_batches}) - Batch Loss: {loss.item():.4f}",
                end="",
            )

        avg_loss = total_loss / num_batches

        print(
            f"\rEpoch {epoch + 1}/{epochs} - 100.00% "
            f"({num_batches}/{num_batches}) - Avg Loss: {avg_loss:.4f}"
        )

        epoch_checkpoint_path = os.path.join(
            model_dir,
            f"checkpoint_epoch_{epoch + 1:03d}.pt",
        )

        save_checkpoint(
            path=epoch_checkpoint_path,
            model=model,
            optimizer=optimizer,
            vocab_size=vocab_size,
            seq_len=seq_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
            mask_id=mask_id,
        )

        print(f"\nSaved epoch checkpoint to {epoch_checkpoint_path}")

    # =========================
    # SAVE FINAL CHECKPOINT
    # =========================

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        mask_id=mask_id,
    )

    print(f"Saved final checkpoint to {checkpoint_path}")

    if not is_resuming:
        print(f"Saved config to {model_dir}/config.json")
        print(f"Saved tokenizer profile to {model_dir}/tokenizer_profile.json")
    else:
        print(f"Kept existing config at {model_dir}/config.json")
        print(f"Kept existing tokenizer profile at {model_dir}/tokenizer_profile.json")