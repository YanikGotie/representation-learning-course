import argparse
import ast
import os

import yaml

import config
from model import AutoEncoder, EncoderClassifier, SupervisedEncoder

# One name from this tuple. config.MODEL_NAME (usually set in the run's config file) is the default.
# pretrain: AutoEncoder, denoising when config.CORRUPTION is not "none".
# pretrain_supervise_encoder: SupervisedEncoder on all labels, the supervised reference encoder.
# knn, linear_probe, finetune: one protocol on the encoder in config.ENCODER_PATH. evaluate: all of them.
# tune: Optuna search (tune.py). analyze: report figures (analyze.py).
MODES = (
    "pretrain",
    "pretrain_supervise_encoder",
    "knn",
    "linear_probe",
    "finetune",
    "evaluate",
    "tune",
    "analyze",
)
DATASETS = ("tiny_imagenet", "cifar100")


def _parse_value(text):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def load_run_config(path, overrides=()):
    """Copy a run's YAML file, then any --set KEY=VALUE, onto the config module.

    Only names that already exist in config.py are accepted, so a typo fails loudly
    instead of being silently ignored.
    """
    values = {}
    if path:
        with open(path) as f:
            values.update(yaml.safe_load(f) or {})
    for item in overrides:
        key, value = item.split("=", 1)
        values[key] = _parse_value(value)
    for key, value in values.items():
        if not hasattr(config, key) or not key.isupper():
            raise KeyError(f"{key!r} is not a name in config.py")
        setattr(config, key, value)
    # YAML keys of a dict like FINETUNE_EPOCHS can arrive as strings from --set
    config.FINETUNE_EPOCHS = {float(k): int(v) for k, v in config.FINETUNE_EPOCHS.items()}


def save_config(path):
    """Write every UPPERCASE name of the config module, so a run can be traced to its settings."""
    values = {k: getattr(config, k) for k in dir(config) if k.isupper()}
    values = {k: (list(v) if isinstance(v, tuple) else v) for k, v in values.items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(values, f, sort_keys=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Train one Part A run.")
    parser.add_argument("--config", help="The run's own YAML file in configs/. Overrides config.py.")
    parser.add_argument("--mode", choices=MODES, help="Overrides MODEL_NAME from the config file.")
    parser.add_argument("--dataset", choices=DATASETS, help="Overrides DATASET from the config file.")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Override single names, e.g. --set NUM_EPOCHS=2 TRAIN_PER_CLASS=20 for a quick check.")
    args = parser.parse_args()
    load_run_config(args.config, args.set)
    if args.mode:
        config.MODEL_NAME = args.mode
    if args.dataset:
        config.DATASET = args.dataset
    if config.TAG is None:
        config.TAG = config.RUN_NAME
    if config.LATENT_DIM_FROM:
        import torch
        state_dict = torch.load(config.LATENT_DIM_FROM, map_location="cpu", weights_only=True)
        config.LATENT_DIM = state_dict["5.weight"].shape[0]  # the bottleneck Linear: [latent_dim, 16384]
    return config.MODEL_NAME


def build_model(mode, num_classes, max_epochs, encoder_path=None, in_features=None, feat_mean=None, feat_std=None,
                probe_weight_decay=0.0):
    """Map a mode to its LightningModule. num_classes and max_epochs depend on the dataset and label fraction."""
    if mode == "pretrain":
        return AutoEncoder(
            learning_rate=config.LEARNING_RATE,
            latent_dim=config.LATENT_DIM,
            corruption=config.CORRUPTION,
            noise_level=config.NOISE_LEVEL,
            mask_patch=config.MASK_PATCH,
            weight_decay=config.WEIGHT_DECAY,
            max_epochs=max_epochs,
        )

    if mode == "pretrain_supervise_encoder" or (mode == "finetune" and encoder_path == "random"):
        # The supervised baseline: the same encoder and head, trained from scratch with the same recipe.
        return SupervisedEncoder(
            learning_rate=config.FINETUNE_LEARNING_RATE,
            num_classes=num_classes,
            latent_dim=config.LATENT_DIM,
            weight_decay=config.FINETUNE_WEIGHT_DECAY,
            max_epochs=max_epochs,
        )

    if mode == "finetune":
        return EncoderClassifier(
            learning_rate=config.FINETUNE_LEARNING_RATE,
            num_classes=num_classes,
            latent_dim=config.LATENT_DIM,
            encoder_path=encoder_path,
            freeze_encoder=False,
            weight_decay=config.FINETUNE_WEIGHT_DECAY,
            max_epochs=max_epochs,
        )

    if mode == "linear_probe":
        # Trained on features the frozen encoder computed once (see evaluation.train_linear_probe).
        return EncoderClassifier(
            learning_rate=config.PROBE_LEARNING_RATE,
            num_classes=num_classes,
            latent_dim=config.LATENT_DIM,
            encoder_path=encoder_path,
            freeze_encoder=True,
            weight_decay=probe_weight_decay,  # evaluation.train_linear_probe tries each of PROBE_WEIGHT_DECAYS
            max_epochs=max_epochs,
            warmup_epochs=0,
            augment=False,
            in_features=in_features,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )

    raise ValueError(f"Mode {mode!r} has no model. Choose from {MODES}")
