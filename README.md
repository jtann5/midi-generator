# midi-generator

PyTorch-based symbolic music generation project for training Transformer models on MIDI datasets and generating new MIDI files from either a BOS token or a seed MIDI clip.

The repository supports two training setups:

- `MAESTRO v3.0.0` for solo piano generation
- `YM2413-MDB v1.0.0` for multi-instrument / game-music-style generation

## What This Repo Does

At a high level, the codebase does four things:

1. Finds MIDI files from a dataset folder.
2. Tokenizes those MIDI files with a MidiTok `REMI` and `REMI+` tokenizer profiles.
3. Trains a causal Transformer to predict the next token in a sequence.
4. Loads saved checkpoints to generate new MIDI outputs and logs.

## Requirements

Python dependencies are listed in [requirements.txt](requirements.txt):

- `numpy`
- `torch`
- `miditok`

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets

The code expects datasets inside the repo under `datasets/`.

### MAESTRO

Source:

- [MAESTRO Dataset](https://magenta.withgoogle.com/datasets/maestro)

Expected local path:

```text
datasets/maestro-v3.0.0
```

This is the path hard-coded in [train_maestro.py](train_maestro.py).

### YM2413-MDB

Source:

- [YM2413-MDB Dataset](https://zenodo.org/records/6566363)

Expected local dataset root:

```text
datasets/YM2413-MDB-v1.0.0
```

The trainer specifically reads MIDI files from:

```text
datasets/YM2413-MDB-v1.0.0/midi/adjust_tempo_remove_delayed_inst
```

This is the path hard-coded in [train_ym2413.py](train_ym2413.py).

## Where To Put Datasets

Place downloaded datasets like this:

```text
midi-generator/
├── datasets/
│   ├── maestro-v3.0.0/
│   │   ├── maestro-v3.0.0.json
│   │   ├── 2004/
│   │   ├── 2006/
│   │   └── ...
│   └── YM2413-MDB-v1.0.0/
│       ├── midi/
│       │   ├── adjust_tempo/
│       │   ├── adjust_tempo_remove_delayed_inst/
│       │   └── vgmplay_log_to_midi/
│       ├── clean_vgm/
│       ├── original_vgms/
│       └── ...
```

The training code recursively scans for `.mid` and `.midi` files under the configured dataset directory.

## Where To Put Models

Saved and downloaded model artifacts belong under `models/`.

Each model should live in its own folder:

```text
models/<model_name>/
├── checkpoint.pt
├── checkpoint_epoch_001.pt
├── checkpoint_epoch_002.pt
├── ...
├── config.json
└── tokenizer_profile.json
```

The generation scripts only treat a model directory as valid if it contains:

- `config.json`
- `tokenizer_profile.json`
- `checkpoint.pt` or one or more `checkpoint_epoch_*.pt` files

If you download or move a trained model into this repo, put it under `models/<model_name>/` in that format.

## Where Generated Samples Go

Generated MIDI files are written under `samples/<model_name>/`.

Example:

```text
samples/maestro_piano_remi_small_seq256_v1/
├── random_001.mid
├── random_001.txt
├── seeded_001.mid
└── seeded_001.txt
```

The `.txt` files are generation logs written next to the MIDI output.

## How To Run The Code

### Train on MAESTRO

```bash
python train_maestro.py
```

Defaults:

- dataset path: `datasets/maestro-v3.0.0`
- tokenizer profile: `piano_remi`
- default model profile: `small_seq256`
- default generation profile: `balanced`
- default max files: `500`
- default model name: `maestro_piano_remi_<model_profile>_v1`

Example with overrides:

```bash
python3 train_maestro.py \
  --model-profile medium_seq256 \
  --generation-profile stable \
  --max-files 1000 \
  --model-name maestro_custom_run_v1
```

### Train on YM2413-MDB

```bash
python3 train_ym2413.py
```

Defaults:

- dataset path: `datasets/YM2413-MDB-v1.0.0/midi/adjust_tempo_remove_delayed_inst`
- tokenizer profile: `multi_instrument_remi`
- default model profile: `small_seq256`
- default generation profile: `balanced`
- default max files: `500`
- default model name: `ym2413_multi_remi_<model_profile>_v1`

### Resume Training

Resume from a saved checkpoint:

```bash
python3 train_maestro.py \
  --resume-checkpoint models/maestro_piano_remi_small_seq256_v1/checkpoint_epoch_012.pt
```

You can optionally override the total epoch target while resuming:

```bash
python3 train_maestro.py \
  --resume-checkpoint models/maestro_piano_remi_small_seq256_v1/checkpoint_epoch_012.pt \
  --epochs 30
```

In resume mode, the saved `config.json` is treated as the source of truth for the run.

### Generate Random MIDI

```bash
python3 generate_random.py
```

Workflow:

1. Choose a model from `models/`.
2. Choose a checkpoint from that model folder.
3. Accept or override generation settings.
4. Choose an output filename.
5. The script writes a MIDI file and a text log under `samples/<model_name>/`.

### Generate From A Seed MIDI

```bash
python3 generate_with_seed.py
```

Workflow:

1. Choose a model from `models/`.
2. Choose a checkpoint.
3. Enter a seed MIDI path, or press Enter to randomly pick one from the model's saved dataset path in `config.json`.
4. Choose seed length and generation settings.
5. The script writes a MIDI file and log under `samples/<model_name>/`.

## Training CLI Options

The shared training runner in [src/training_runner.py](src/training_runner.py) supports these arguments:

```text
--model-name
--model-profile
--generation-profile
--max-files
--overwrite
--resume-checkpoint
--seq-len
--chunk-stride
--batch-size
--epochs
--lr
--d-model
--nhead
--num-layers
--ff-dim
--dropout
```

Model profiles are defined in [src/model_profiles.py](src/model_profiles.py).

Tokenizer profiles are defined in [src/tokenizer_profiles.py](src/tokenizer_profiles.py).

## Codebase Structure

### Top Level Files

- [train_maestro.py](train_maestro.py)
  Entry point for training on MAESTRO. Calls the shared training runner with the piano tokenizer profile and MAESTRO dataset path.

- [train_ym2413.py](train_ym2413.py)
  Entry point for training on YM2413-MDB. Calls the shared training runner with the multi-instrument tokenizer profile and YM2413 dataset path.

- [generate_random.py](generate_random.py)
  Interactive generation script that starts from the BOS token and produces a new random MIDI sample.

- [generate_with_seed.py](generate_with_seed.py)
  Interactive generation script that tokenizes a seed MIDI file, then continues generation from those seed tokens.


### `src/`

- [src/training_runner.py](src/training_runner.py)
  Main training pipeline. Parses CLI args, resolves profiles, prepares model directories, tokenizes the dataset, builds training chunks, trains the Transformer, and saves checkpoints/config files.

- [src/dataset.py](src/dataset.py)
  Dataset utilities:
  finds MIDI files, tokenizes them, slices token sequences into `(X, Y)` next-token training chunks, and wraps them in a PyTorch `Dataset`.

- [src/transformer.py](src/transformer.py)
  Defines the model:
  token embedding, sinusoidal positional encoding, causal `TransformerEncoder`, and output projection layer.

- [src/checkpoint.py](src/checkpoint.py)
  Saves checkpoints and reconstructs a `MusicTransformer` from a saved checkpoint file.

- [src/config.py](src/config.py)
  Creates and persists `config.json`, loads configs for later generation/resume, and prepares model directories safely.

- [src/tokenizer_profiles.py](src/tokenizer_profiles.py)
  Defines tokenizer presets such as `piano_remi` and `multi_instrument_remi`, builds MidiTok tokenizers from those presets, and saves/loads `tokenizer_profile.json`.

- [src/tokenizer_utils.py](src/tokenizer_utils.py)
  Utility helpers for tokenizer vocab lookup, special token IDs, and converting a seed MIDI into input token IDs.

- [src/model_profiles.py](src/model_profiles.py)
  Central registry for model-size presets and generation presets.

- [src/generation.py](src/generation.py)
  Core token sampling logic:
  autoregressive generation, top-k / top-p filtering, repetition penalty handling, special-token stripping, and MIDI decoding.

- [src/generation_cli.py](src/generation_cli.py)
  Interactive generation UI in the terminal:
  lists models, lists checkpoints, prompts for sampling settings, manages output naming, and writes generation logs.

- [src/device.py](src/device.py)
  Chooses the best available Torch device in this order: `mps`, `cuda`, then `cpu`.

### Data And Artifact Folders

- `datasets/`
  Expected home for raw MIDI datasets.

- `models/`
  Saved training runs and checkpoints.

- `samples/`
  Generated MIDI outputs and text generation logs.

## Training Pipeline

The training flow in [src/training_runner.py](src/training_runner.py) is:

1. Parse CLI arguments.
2. Resolve dataset path, tokenizer profile, model profile, and generation profile.
3. Create or reuse `models/<model_name>/`.
4. Save `config.json` and `tokenizer_profile.json` for new runs.
5. Find MIDI files from the dataset directory.
6. Tokenize each MIDI file.
7. Add `BOS` and `EOS` tokens.
8. Slice token streams into overlapping training chunks.
9. Train the Transformer with cross-entropy next-token prediction.
10. Save `checkpoint_epoch_XXX.pt` after every epoch.
11. Save final `checkpoint.pt`.

## Generated Model Folder Contents

For a new training run, the code creates:

- `models/<model_name>/config.json`
- `models/<model_name>/tokenizer_profile.json`
- `models/<model_name>/checkpoint_epoch_001.pt`
- `models/<model_name>/checkpoint_epoch_002.pt`
- ...
- `models/<model_name>/checkpoint.pt`

`config.json` stores:

- dataset name and path
- tokenizer profile name
- model profile and generation profile names
- training hyperparameters
- model architecture settings
- generation defaults
- parameter counts
- notes
