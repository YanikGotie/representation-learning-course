"""Part A entry point, grown from lec5's train.py. Every run is this file plus its own config:

    python train.py --config configs/pretrain_dae.yaml
    python train.py --config configs/pretrain_dae.yaml --set NUM_EPOCHS=2 TRAIN_PER_CLASS=20   # quick check

lec5 ran one fit -> validate -> test per call and saved the encoder. That is still what
pretrain, pretrain_supervise_encoder and finetune do (see fit_model). Part A adds:
  * model selection by validation k-NN during pretraining (callbacks.KNNMonitor), not by the loss
  * the evaluation protocol (knn, linear_probe, finetune) on Tiny-ImageNet and CIFAR-100,
    at 1/10/100% of the labels, with every number appended to results/protocol.csv
  * tune (tune.py) and analyze (analyze.py)
"""
import json
import os
import time

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import config
from callbacks import MyPrintingCallback, EarlyStopping, ColabProgressBar, EpochMetricsCallback, KNNMonitor
from datamodule import TinyImageNetDataModule, balanced_subset
from evaluation import extract_features, knn_eval, per_class_accuracy, probe_predict, topk_acc, train_linear_probe
from utils import build_model, parse_args, save_config

torch.set_float32_matmul_precision("medium")  # to make lightning happy
RUN_ID = time.strftime("%Y-%m-%d_%H-%M-%S")


# --------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------
def compute_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_trainer(max_epochs, logger=None, precision=None, monitor=None, ckpt_dir=None, callbacks=(),
                 enable_progress_bar=True):
    """The lec5 Trainer, plus a ModelCheckpoint when `monitor` is given.

    ModelCheckpoint keeps the epoch with the best validation score (k-NN accuracy for pretraining,
    accuracy for classifiers) and the last epoch (save_last=True), so an interrupted run resumes.
    """
    cbs = [MyPrintingCallback(), EpochMetricsCallback(), *callbacks]
    if enable_progress_bar:
        cbs.append(ColabProgressBar(refresh_rate=20))
    if logger:
        cbs.append(LearningRateMonitor(logging_interval="epoch"))
    if config.EARLY_STOPPING_PATIENCE:
        cbs.append(EarlyStopping(monitor="val_loss", patience=config.EARLY_STOPPING_PATIENCE))
    if monitor:
        cbs.append(ModelCheckpoint(dirpath=ckpt_dir, filename="best", monitor=monitor, mode="max", save_last=True))
    return pl.Trainer(
        logger=logger if logger is not None else False,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=max_epochs,
        precision=precision or config.PRECISION,
        log_every_n_steps=20,
        callbacks=cbs,
        enable_checkpointing=monitor is not None,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=False,
    )


def make_datamodule(dataset, **kw):
    dm = TinyImageNetDataModule(dataset=dataset, data_dir=config.DATA_DIR, seed=config.SEED,
                                source=config.DATA_SOURCE if dataset == "tiny_imagenet" else None, **kw)
    dm.prepare_data()
    dm.setup()
    return dm


def run_dir(*parts):
    path = os.path.join(config.CHECKPOINT_DIR, config.RUN_NAME, *parts)
    os.makedirs(path, exist_ok=True)
    return path


def append_rows(csv_path, rows, key_cols):
    """Append rows to a results CSV, replacing older rows with the same key (a re-run overwrites)."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    new = pd.DataFrame(rows)
    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path)
        if all(c in old.columns for c in key_cols):
            keys = set(map(tuple, new[key_cols].astype(str).values))
            old = old[[tuple(r) not in keys for r in old[key_cols].astype(str).values]]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(csv_path, index=False)


def fit_model(mode, dm, max_epochs, monitor, ckpt_dir, logger, callbacks=(), encoder_path=None, verbose=True):
    """lec5's run(): build the model for `mode`, fit, then validate and test.

    The model that is validated, tested and saved is the epoch ModelCheckpoint kept (the best
    validation score), not simply the last one.
    """
    model = build_model(mode, dm.num_classes, max_epochs, encoder_path=encoder_path)
    trainer = make_trainer(max_epochs, logger=logger, monitor=monitor, ckpt_dir=ckpt_dir, callbacks=callbacks,
                           enable_progress_bar=verbose)
    if verbose:
        print(model)
        print(f"Training mode is: {mode}, on {dm.dataset} for {max_epochs} epochs, batch size {dm.batch_size}, "
              f"n_train={len(dm.train_y)}")
    last = os.path.join(ckpt_dir, "last.ckpt")
    t0 = time.time()
    trainer.fit(model, dm, ckpt_path=last if os.path.exists(last) else None)  # resumes an interrupted run
    minutes = (time.time() - t0) / 60

    ckpt_cb = next(c for c in trainer.callbacks if isinstance(c, ModelCheckpoint))
    best_path = ckpt_cb.best_model_path or last
    best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["state_dict"])
    val = trainer.validate(model, dm, verbose=False)[0]
    test = trainer.test(model, dm, verbose=False)[0]
    return model, {"val": val, "test": test, "best_epoch": int(best_ckpt["epoch"]), "train_minutes": minutes}


def encoder_file():
    """Where a pretrain run writes its encoder: ENCODER_PATH, or checkpoints/<RUN_NAME>/encoder.pt."""
    return config.ENCODER_PATH or os.path.join(run_dir(), "encoder.pt")


# --------------------------------------------------------------------------------------
# pretrain: the (denoising) autoencoder
# --------------------------------------------------------------------------------------
def run_pretrain(dm=None, logger_name=None, verbose=True):
    """Train an AutoEncoder on unlabelled Tiny-ImageNet, keep the epoch with the best validation
    k-NN accuracy, and save its encoder. tune.py calls this for every trial."""
    pl.seed_everything(config.SEED, workers=True)  # same initialization and data order -> fair comparison
    dm = dm or make_datamodule(config.DATASET, batch_size=config.BATCH_SIZE, train_per_class=config.TRAIN_PER_CLASS,
                               drop_last=True)
    out_dir = run_dir()
    save_config(os.path.join(out_dir, "config.yaml"))
    logger = TensorBoardLogger(config.LOG_DIR, name=logger_name or config.RUN_NAME, version="pretrain",
                               default_hp_metric=False)
    knn = KNNMonitor(dm, every_n_epochs=config.KNN_EVERY_N_EPOCHS, k=config.KNN_K,
                     bank_per_class=config.KNN_BANK_PER_CLASS)
    model, res = fit_model("pretrain", dm, config.NUM_EPOCHS, "val_knn_top1", out_dir, logger, callbacks=[knn],
                           verbose=verbose)

    # Keep the encoder. The decoder is not saved.
    path = encoder_file()
    torch.save(model.encoder.state_dict(), path)

    # Final numbers for the selected epoch, with the FULL train split as the k-NN memory bank.
    model = model.to(compute_device())
    knn_full = knn_eval(model.encoder, dm.bank_x, dm.bank_y, dm.val_x, dm.val_y, dm.num_classes, k=config.KNN_K)
    result = {
        "run_name": config.RUN_NAME, "latent_dim": config.LATENT_DIM, "corruption": config.CORRUPTION,
        "noise_level": config.NOISE_LEVEL, "learning_rate": config.LEARNING_RATE, "weight_decay": config.WEIGHT_DECAY,
        "epochs": config.NUM_EPOCHS, "train_per_class": config.TRAIN_PER_CLASS, "best_epoch": res["best_epoch"],
        "val_knn_top1": knn_full["top1"], "val_knn_top5": knn_full["top5"],
        "val_recon_mse": res["val"]["val_recon_mse"], "val_denoise_mse": res["val"]["val_loss"],
        "test_recon_mse": res["test"]["test_recon_mse"], "train_minutes": res["train_minutes"],
        "encoder_path": path,
    }
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump({"result": result, "history": knn.history}, f, indent=1)
    append_rows(os.path.join(config.RESULTS_DIR, "pretrain_runs.csv"), [result], ["run_name"])
    if verbose:
        print(json.dumps(result, indent=1))
    return result


# --------------------------------------------------------------------------------------
# pretrain_supervise_encoder: the supervised reference encoder
# --------------------------------------------------------------------------------------
def run_supervise_encoder():
    """SupervisedEncoder on 100% of the Tiny-ImageNet labels. Its saved encoder is evaluated
    frozen (k-NN and linear probe) as the supervised upper reference."""
    pl.seed_everything(config.SEED, workers=True)
    dm = make_datamodule(config.DATASET, batch_size=config.FINETUNE_BATCH_SIZE, label_fraction=1.0)
    out_dir = run_dir()
    save_config(os.path.join(out_dir, "config.yaml"))
    logger = TensorBoardLogger(config.LOG_DIR, name=config.RUN_NAME, version="pretrain_supervise_encoder",
                               default_hp_metric=False)
    model, res = fit_model("pretrain_supervise_encoder", dm, config.FINETUNE_EPOCHS[1.0], "val_accuracy",
                           out_dir, logger)
    torch.save(model.encoder.state_dict(), encoder_file())
    print(json.dumps({k: res[k] for k in ("best_epoch", "test")}, indent=1))
    return res["test"]["test_accuracy"]


# --------------------------------------------------------------------------------------
# knn / linear_probe / finetune / evaluate: the assignment's protocol
# --------------------------------------------------------------------------------------
def run_evaluate(protocols):
    """Run each protocol for the encoder in ENCODER_PATH on every dataset in EVAL_DATASETS.

    Rows go to results/protocol.csv (test top-1 / top-5, and val for reference), keyed by
    TAG, dataset, protocol and label fraction, so a re-run replaces its own rows.
    FEATURE_LAYER="conv" evaluates the pooled last conv map instead of h; those rows are
    tagged "<TAG>@conv". Fine-tuning always trains a head on h.
    """
    from model import load_encoder

    tag, layer = config.TAG, config.FEATURE_LAYER
    if layer != "h":
        if "finetune" in protocols:
            raise ValueError("FEATURE_LAYER only applies to knn and linear_probe; finetune always uses h")
        tag = f"{tag}@{layer}"
    dev = compute_device()
    csv = os.path.join(config.RESULTS_DIR, "protocol.csv")
    key = ["tag", "dataset", "protocol", "label_fraction"]
    per_class_dir = os.path.join(config.RESULTS_DIR, "per_class")
    os.makedirs(per_class_dir, exist_ok=True)
    save_config(os.path.join(run_dir(), "config.yaml"))
    rows = []

    def record(row):
        rows.append(row)
        append_rows(csv, [row], key)
        print(row)

    for ds in config.EVAL_DATASETS:
        dm = make_datamodule(ds, batch_size=config.FINETUNE_BATCH_SIZE)
        pl.seed_everything(config.SEED)
        encoder = load_encoder(config.ENCODER_PATH, config.LATENT_DIM).to(dev)

        # ---- 1. k-NN (k=20, cosine, similarity-weighted vote; no training) ----
        if "knn" in protocols:
            val = knn_eval(encoder, dm.bank_x, dm.bank_y, dm.val_x, dm.val_y, dm.num_classes, k=config.KNN_K,
                           layer=layer)
            test = knn_eval(encoder, dm.bank_x, dm.bank_y, dm.test_x, dm.test_y, dm.num_classes, k=config.KNN_K,
                            layer=layer)
            np.save(os.path.join(per_class_dir, f"{tag}_{ds}_knn.npy"),
                    per_class_accuracy(test["pred"], dm.test_y, dm.num_classes))
            record(dict(tag=tag, dataset=ds, protocol="knn", label_fraction=1.0, val_top1=val["top1"],
                        val_top5=val["top5"], test_top1=test["top1"], test_top5=test["top5"]))

        # ---- 2. Linear probe: EncoderClassifier with the encoder frozen ----
        if "linear_probe" in protocols:
            bank_f = extract_features(encoder, dm.bank_x, layer=layer)
            val_f = extract_features(encoder, dm.val_x, layer=layer)
            test_f = extract_features(encoder, dm.test_x, layer=layer)
            y_all = np.full(dm.full_train_idx.max() + 1, -1)
            y_all[dm.full_train_idx] = dm.bank_y.numpy()
            for frac in config.LABEL_FRACTIONS:
                sub = balanced_subset(dm.full_train_idx, y_all, fraction=frac, seed=config.SEED)
                pos = torch.from_numpy(np.searchsorted(dm.full_train_idx, sub))
                logger = TensorBoardLogger(config.LOG_DIR, name=tag, version=f"{ds}_linear_probe_{frac}",
                                           default_hp_metric=False)
                probe, wd, wd_scores = train_linear_probe(bank_f[pos], dm.bank_y[pos], val_f, dm.val_y,
                                                          dm.num_classes, logger=logger)
                val_s, test_s = probe_predict(probe, val_f), probe_predict(probe, test_f)
                if frac == 1.0:
                    np.save(os.path.join(per_class_dir, f"{tag}_{ds}_linear.npy"),
                            per_class_accuracy(test_s.argmax(1), dm.test_y, dm.num_classes))
                record(dict(tag=tag, dataset=ds, protocol="linear_probe", label_fraction=frac,
                            val_top1=topk_acc(val_s, dm.val_y, 1), val_top5=topk_acc(val_s, dm.val_y, 5),
                            test_top1=topk_acc(test_s, dm.test_y, 1), test_top5=topk_acc(test_s, dm.test_y, 5),
                            info=json.dumps({"weight_decay": wd, "val_by_wd": wd_scores})))

        # ---- 3. Fine-tuning (all weights) on class-balanced label subsets ----
        # ENCODER_PATH="random" gives the SupervisedEncoder baseline, trained from scratch.
        if "finetune" in protocols:
            for frac in config.LABEL_FRACTIONS:
                pl.seed_everything(config.SEED, workers=True)
                dm_f = make_datamodule(ds, batch_size=config.FINETUNE_BATCH_SIZE, label_fraction=frac)
                epochs = config.FINETUNE_EPOCHS[float(frac)]
                logger = TensorBoardLogger(config.LOG_DIR, name=tag, version=f"{ds}_finetune_{frac}",
                                           default_hp_metric=False)
                _, res = fit_model("finetune", dm_f, epochs, "val_accuracy", run_dir(f"{ds}_finetune_{frac}"),
                                   logger, encoder_path=config.ENCODER_PATH)
                record(dict(tag=tag, dataset=ds, protocol="finetune", label_fraction=frac,
                            val_top1=res["val"]["val_accuracy"], val_top5=res["val"]["val_top5"],
                            test_top1=res["test"]["test_accuracy"], test_top5=res["test"]["test_top5"],
                            info=json.dumps({"epochs": epochs, "best_epoch": res["best_epoch"],
                                             "n_train": len(dm_f.train_y)})))
    return rows


# --------------------------------------------------------------------------------------
def run(mode):
    """Train and evaluate one experiment. mode: config.MODEL_NAME (from the run's config file), or --mode."""
    pl.seed_everything(config.SEED)  # same initialization and data order -> fair comparison
    if mode == "pretrain":
        return run_pretrain()
    if mode == "pretrain_supervise_encoder":
        return run_supervise_encoder()
    if mode in config.EVAL_MODES:
        return run_evaluate(config.PROTOCOLS if mode == "evaluate" else [mode])
    if mode == "tune":
        from tune import run_tune
        return run_tune()
    if mode == "analyze":
        from analyze import run_analyze
        return run_analyze()
    raise ValueError(f"Unknown mode {mode!r}")


if __name__ == "__main__":
    mode = parse_args()
    print(f"mode={mode} run_name={config.RUN_NAME} dataset={config.DATASET}")
    value = run(mode)

    print("\nTest results:")
    if mode == "pretrain":
        print(f"  {mode:<28} ({config.DATASET}): val k-NN top-1 {value['val_knn_top1']:.4f}, "
              f"clean reconstruction MSE {value['val_recon_mse']:.4f}")
    elif mode == "pretrain_supervise_encoder":
        print(f"  {mode:<28} ({config.DATASET}): accuracy {value:.3f}")
    elif mode in config.EVAL_MODES:
        for r in value:
            print(f"  {r['tag']:<12} {r['dataset']:<14} {r['protocol']:<13} {r['label_fraction']:<5} "
                  f"top-1 {r['test_top1']:.3f}  top-5 {r['test_top5']:.3f}")
